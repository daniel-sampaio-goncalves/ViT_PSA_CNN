from typing import Callable
from timeit import default_timer as timer

import torch
from torch import nn
from tqdm.auto import tqdm

from utils.utils import save_model

###################################
def accuracy_fn_binary(y_true:torch.Tensor, _y_preds:torch.Tensor) -> torch.Tensor:
    """
    Compute binary classification accuracy as a percentage.

    Args:
        y_true (torch.Tensor): Ground truth binary labels either 0 or 1.
        _y_preds (torch.Tensor): Predicted probabilities scores between {0, 1} (e.g. after sigmoid)

    Returns:
        float: Accuracy as a percentage as a plain Python float
    """
    _y_preds = (_y_preds >= 0.5).int()
    return (_y_preds == y_true).float().mean() * 100


def print_train_time(
            start: float,
            end: float,
            device: torch.Device | None = None
                    ):
    """
    Print and return the elapsed training time.

    Args:
        start (float): Start time, typically from ``time.time()``.
        end (float): End time, typically from ``time.time()``.
        device (troch.Device|None): Optionally, the device used for training.

    Returns:
        None: 
            prints the elapsed time in seconds.
    """
    total_time=end-start
    print(f'Train time on {device}: {total_time:.3f}')
    return total_time

def mask_scale_linear(epoch:int, end_anneal_epoch:int, start:float=0.5, end:float=0.95, warmup_val:float=0.5, warmup_frac:float=0.2, final_val:float=0.90):
    """
    Linear annealing schedule for how much patch masking  (ratio of masked patches float: {0:1}) is used.

    Args:
        epoch (int): Current epoch
        end_anneal_epoch (int): last epoch that returns `end` and also sets the warm-up length via `warmup_frac`.
        start (float): Starting value (ratio of masked patches {0:1}) of the first ramp epoch.
        end (float): Ending value (ratio of masked patches {0:1}) at `end_anneal_epoch`.
        warmup_val (float): Constant value (ratio of masked patches {0:1}) returned during warm up.
        warmup_frac (float): Fraction of `end_anneal_epoch` that will be served as warm up period.
        final_val (float): Constant value (ratio of masked patches {0:1}) returned after `end_anneal_epoch` onwards.

    Returns:
        float:
         - The % of masked patches between 0:1 for the given epoch.
    """
    if epoch < end_anneal_epoch * warmup_frac:
        return warmup_val
    
    elif epoch<=end_anneal_epoch:
        frac = (epoch - end_anneal_epoch*warmup_frac) / (end_anneal_epoch*((1)-warmup_frac))
        scale= start + frac * (end - start)
        return scale 

    else:
        return  final_val

###Vit Only:
def evaluation_autocast_vit(
          model:nn.Module, 
          test_data_loader:torch.utils.data.DataLoader[dict[str,torch.Tensor]],
          loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],  
          accuracy_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=accuracy_fn_binary, 
          object_index:list[str]=["image","label"]
          ):
    """
    Evaluate the ViT classification module under bf16 autocast.

    Args:
        model (nn.Module): Loaded model on CUDA.
        test_data_loader (DataLoader): A pytorch DataLoader with batches as dictionaries with the keys mapping to tensors given in `object_index`.
        loss_fn (Callable): Classification loss called as `loss_fn(logits, label)`.
        accuracy_fn (Callable): Accuracy metric function expected to return a batch mean, expected to return a tensor to avoid gpu/cpu sync.
        object_index (list[str]): Keys used to access the tensors in each batch returned by `test_data_loader`.

    Returns:
        (mean_loss, mean_accuracy): per-sample means over the whole loader, each batch weighted by its
        size, as 0-d CUDA tensors.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    
    if torch.cuda.is_available():
         device = 'cuda'  
    else:
         raise RuntimeError("CUDA is not available. This code uses CUDA-specific optimizations and cannot run on CPU.")

    model.eval()
    test_loss = torch.zeros((), device=device)
    test_acc = torch.zeros((), device=device)
    n_samples=0
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        
        for batch, loaded_data in enumerate(test_data_loader):
            X, label = loaded_data[object_index[0]], loaded_data[object_index[1]]
            X=X.to(device, non_blocking=True, memory_format=torch.channels_last)

            label=label.to(device, dtype=torch.float32, non_blocking=True)
            test_label=model(X, return_vit_class=True)
            loss=loss_fn(test_label, label)
            
            acc=accuracy_fn(label, (torch.sigmoid(test_label)).to(dtype=torch.float32))

            # scale by n samples used
            n = label.shape[0]
            n_samples += n

            test_loss+= loss*n
            test_acc += acc*n

        # average with total sampels used
        test_loss/=n_samples
        test_acc/=n_samples

    return test_loss,test_acc


def train_autocast_vit(
          model:nn.Module, 
          train_data_loader: torch.utils.data.DataLoader[dict[str,torch.Tensor]], 
          optimizer:torch.optim.Optimizer,
          loss_fn:Callable[[torch.Tensor, torch.Tensor],torch.Tensor],
          accuracy_fn:Callable[[torch.Tensor, torch.Tensor],torch.Tensor]=accuracy_fn_binary, 
          scheduler: torch.optim.lr_scheduler.LRScheduler|None=None,
          grad_clip:bool=False, 
          max_grad_norm:float = 1.0, 
          object_index:list[str]=["image","label"],
          mask_val: float|torch.Tensor=0.0)-> tuple[torch.Tensor,torch.Tensor]:
    """
    Runs one training epoch under bf16 autocast and returns a sample weighted mean loss and accuracy.
    
    Args:
        model (nn.Module): model, must already be on CUDA.
        train_data_loader (torch.utils.data.DataLoader[dict[str,torch.Tensor]]): A pytorch DataLoader with batches as dictionaries with the keys mapping to tensors given in `object_index`.
        optimizer (optim.Optimizer): pytorch optimizer stepped once per batch.
        loss_fn (Callable): classification loss `loss_fn(logits, label)` (e.g. BCE with logits).
        accuracy_fn (Callable): accuracy metric function with probabilities as arguments.
        scheduler (lr_scheduler.LRScheduler) | None: Optionally, a per batch scheduler, defaults to None.
        grad_clip (bool): Defaults to False. If True, clips the global gradient norm to `max_grad_norm`.
        max_grad_norm (float): Clipping threshold, used only when `grad_clip` is True.
        object_index (list[str]): Keys used to access the tensors in each batch returned by `train_data_loader`.
        mask_val (float|torch.Tensor): Fraction of patches to be masked during training. Ratio, float values between 0 and 1.
        
    Returns:
        tuple(train_loss[torch.Tensor], train_accuracy[torch.Tensor].

    Raises:
        RuntimeError: If CUDA is not available.
    """

    if torch.cuda.is_available():
         device = 'cuda'  
    else:
         raise RuntimeError("CUDA is not available. This code uses CUDA-specific optimizations and cannot run on CPU.")

    train_loss = torch.zeros((), device=device)
    train_acc = torch.zeros((), device=device)
    n_samples = torch.zeros((), device=device)

    model.train()
    for batch, loaded_data in enumerate(train_data_loader):
        X, label = loaded_data[object_index[0]], loaded_data[object_index[1]]
        X=X.to(device, non_blocking=True, memory_format=torch.channels_last)
        label=label.to(device, dtype=torch.float32, non_blocking=True)


        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # trains in bflat16 lower memory and faster less precision though - no gradscaling necessary compared
            # forward
            pred_label=model(X, return_vit_class=True, mask_ratio=mask_val)
            # loss
            loss = loss_fn(pred_label, label)
            
        
        # backward

        loss.backward()

        if grad_clip:
            # Gradient clipping (global norm)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm
                )
        # optimize step
        optimizer.step()

        # zerograd
        optimizer.zero_grad(set_to_none=True) # it deallocates grads instead of writing zeros into them, which is faster and uses less memory.

        if scheduler:
            scheduler.step()
        
        # track samples
        n = label.shape[0]
        n_samples += n

        with torch.no_grad():        
            # Convert each batch mean into a sample-weighted total.
            # alpha=n performs the scaling as part of an in-place addition,
            # avoiding a separate x * n operation and intermediate tensor.
            train_acc.add_(accuracy_fn(label, torch.sigmoid(pred_label).float()).detach(), alpha=n)
            train_loss.add_(loss.detach(), alpha=n)

        
        if batch == 0:
            print(f"Batch: {batch}/{len(train_data_loader)}")
        if (batch + 1) % 100 == 0:
            print(f"Batch: {batch+1}/{len(train_data_loader)}")
        if (batch+1) == len(train_data_loader):
            print(f'Batch: {batch+1}/{len(train_data_loader)}')

    # Divide n_sampels 
    train_loss /=n_samples
    train_acc /=n_samples

    return train_loss,train_acc

def model_train_vit(
          model:nn.Module, 
          epochs:int, 
          optimizer:torch.optim.Optimizer, 
          loss_fn:Callable[[torch.Tensor, torch.Tensor],torch.Tensor], 
          train_data_loader:torch.utils.data.DataLoader[dict[str,torch.Tensor]], 
          test_data_loader:torch.utils.data.DataLoader[dict[str,torch.Tensor]], 
          lr_scheduler:None|torch.optim.lr_scheduler.LRScheduler=None, 
          grad_clip:float|None=None, 
          end_anneal:int=10, 
          mask_anneal:bool=True,
          accuracy_savepoint:float=80,
          compiled_model:bool=True
          ) -> list[list[int | dict] | dict]:
    """
    Run the full ViT training loop. It launches a per epoch train/eval loop with optional mask annealing 
    and uses checkpointing. Requires CUDA.
    
    Args:
        model (torch.nn.Module): Model to train.
        epochs (int): Number of epochs to run.
        optimizer (torch.optim.Optimizer): A pytorch optimizer
        loss_fn (torch.nn.Module): classification loss `loss_fn(logits, label)` (e.g. BCE with logits).
        train_data_loader (torch.utils.data.DataLoader): A pytorch DataLoader with training data.
        test_data_loader (torch.utils.data.DataLoader): A pytorch DataLoader with test/eval data.
        lr_scheduler (lr_sheduler.LRScheduler|None): Optional, a per batch scheduler used in :func:`train_autocast` after each optimizer step.
        grad_clip (float|None): Gradient norm value to be used as max clip value. None disables clipping. Defaults to None.
        end_anneal (int): Last epoch until annealing value stops chaning. See :func:`mask_scale_linear`
        mask_anneal (bool): If True, uses annealing masking values generated by :func:`mask_scale_linear`. Defaults to True.
        accuracy_savepoint (float): Evaluation accuracy % upon which a model's weights are saved. Checked every loop.
        compiled_model (bool): Needed for saving a checkpoint correctly depending on if a `model` was compiled with `torch.compile()` or not. Passed to :func:save_model.
    
    Returns:
        list[ list[epoch, dict[str, training/test metric]], dict[str, run time] ]: 
            - `[epoch, {metrics_dict}]` entry per epoch, followed by a final `{"train_time": total_time}`
        
    Raises:
        RuntimeError: If CUDA is not available.
    
    """
    train_start_time=timer()
    if torch.cuda.is_available():
         device = 'cuda'  
    else:
         raise RuntimeError("CUDA is not available. This code uses CUDA-specific optimizations and cannot run on CPU.")
    model.to(device)
    epochs_results=[]

    # get the last groups applied lr:
    for param_group in optimizer.param_groups:
                lr=param_group['lr']

    if lr_scheduler:
            print("schedule was applied")
    if grad_clip is not None:
            print(f"gradient clipping was applied with a value of {grad_clip}")
            grad_=True
    else:
            grad_=False
            # grad is false so grad_clip won't be used
            # assigning float for type checking
            grad_clip=1.0


    print("using autocast bf16")

    for epoch in tqdm(range(epochs)):

        if mask_anneal is True:
            mask_val=torch.tensor(
                                  mask_scale_linear(
                                       epoch=epoch, end_anneal_epoch=end_anneal),
                                       device=device
                                       )
        else:
            mask_val= torch.zeros((), device=device)
        print(f"applying {mask_val*100}% of patch masking")
        


        print(f'-----------')
        
        train_loss,train_acc=train_autocast_vit(model, train_data_loader,loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn_binary, scheduler=lr_scheduler, grad_clip=grad_, max_grad_norm=grad_clip, mask_val=mask_val )

        if lr_scheduler:
                    for param_group in optimizer.param_groups:
                        lr=param_group['lr']
        
        test_loss, test_acc=evaluation_autocast_vit(model, test_data_loader, loss_fn=loss_fn, accuracy_fn=accuracy_fn_binary)
        
        
        
        print(f'EPOCH {epoch} || Train loss {train_loss:.4f} | Train acc: {train_acc:.2f}% || Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}% || last sample lr={lr:.3e}')   # type: ignore   
        print("----------------------------")
        epochs_results.append([epoch, {"epochs":epochs, "train_loss":train_loss, "train_acc":train_acc, "test_loss":test_loss, "test_acc":test_acc, "train_time":None}])
        if train_acc>accuracy_savepoint:
                    save_model(model=model,safe_root="",model_name=f"./models/weights/Vit_Pretraining_e{epoch}_chpt",compiled=compiled_model)
                    print("#####################")
    train_end_time=timer()

    total_time=print_train_time(train_start_time, train_end_time, device=device)
    epochs_results.append({"train_time":total_time})
    
    return epochs_results