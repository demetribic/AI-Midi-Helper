# generate.py

from argparse import ArgumentParser
from copy import deepcopy
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import datetime
import bisect
import pretty_midi  # Ensure pretty_midi is installed
import torch.nn.utils.prune as prune
from .generate_utils import validate_output_dir


def safe_mps_operation(operation, device, *tensors, **kwargs):
    """
    Helper function to safely handle MPS operations with CPU fallback.

    Args:
        operation (callable): The PyTorch operation to perform.
        *tensors: Tensor inputs for the operation.
        **kwargs: Additional keyword arguments for the operation.

    Returns:
        torch.Tensor: The result of the operation.
    """
    try:
        return operation(*tensors, **kwargs)
    except Exception as e:
        if device.type == 'mps':
            # Move tensors to CPU, perform operation, move back to MPS
            cpu_tensors = [t.cpu() if torch.is_tensor(t) else t for t in tensors]
            result = operation(*cpu_tensors, **kwargs)
            try:
                return result.to(device)
            except:
                return result
        raise e




def compile_model_safely(model, device):
    """
    Safely compiles the model taking into account device compatibility.

    Args:
        model (torch.nn.Module): The PyTorch model to compile.
        device (torch.device): The device the model is on.

    Returns:
        torch.nn.Module: The compiled model (or original model if compilation is not supported).
    """
    # Skip compilation for MPS devices due to compatibility issues
    if device.type == 'mps':
        print("Skipping torch.compile() for MPS device")
        return model

    try:
        compiled_model = torch.compile(model)
        print("Model compiled successfully with torch.compile()")
        return compiled_model
    except Exception as e:
        print.warning(f"torch.compile() failed: {e}. Continuing with uncompiled model")
        return model


import torch
import torch.nn as nn
import torch.quantization
from torch.nn.utils import prune
from typing import List, Tuple, Optional

def prune_model_weights(
    model: nn.Module,
    amount: float = 0.2,
    excluded_layer_types: Optional[List[type]] = None
) -> nn.Module:
    """
    Prunes model weights globally using L1 unstructured pruning, with configurable layer exclusions.
    
    Args:
        model: The PyTorch model to prune
        amount: Fraction of connections to prune (0.0 to 1.0)
        excluded_layer_types: List of layer types to exclude from pruning (defaults to [nn.Embedding])
    
    Returns:
        The pruned model
    """
    if amount < 0.0 or amount > 1.0:
        raise ValueError("Pruning amount must be between 0.0 and 1.0")
        
    if excluded_layer_types is None:
        excluded_layer_types = [nn.Embedding]
        
    parameters_to_prune: List[Tuple[nn.Module, str]] = []
    
    for name, module in model.named_modules():
        if any(isinstance(module, layer_type) for layer_type in excluded_layer_types):
            continue
            
        if hasattr(module, 'weight') and module.weight is not None:
            parameters_to_prune.append((module, 'weight'))
    
    if not parameters_to_prune:
        print.warning("No eligible parameters found for pruning")
        return model
        
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    print(
        f"Pruned {amount*100:.1f}% of model weights globally "
        f"(excluding {[t.__name__ for t in excluded_layer_types]})"
    )
    return model

def quantize_model(model: nn.Module, target_device: str = "mps") -> nn.Module:
    """
    Applies dynamic quantization to the model with proper error handling and device management.
    
    Args:
        model: The PyTorch model to quantize
        target_device: The device to return the model to after quantization
        
    Returns:
        The quantized model (or original model if quantization fails)
    """
    # Store original device
    original_device = next(model.parameters()).device
    
    try:
        # Move to CPU for quantization
        model = model.cpu()
        
        # Attempt quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Support more layer types
            dtype=torch.qint8,
            inplace=False  # Create a new model instead of modifying in-place
        )
        
        print("Dynamic quantization applied successfully")
        return quantized_model.to(target_device)
        
    except Exception as e:
        print.warning(
            f"Dynamic quantization failed: {e}\n"
            f"Device '{target_device}' may not support quantization. "
            "Returning original model."
        )
        return model.to(original_device)

def remove_weight_norms(model: nn.Module) -> None:
    """
    Safely removes weight normalization from all applicable layers.
    
    Args:
        model: The PyTorch model to modify
    """
    for module in model.modules():
        if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            torch.nn.utils.remove_weight_norm(module)
            
    


def safe_device_transfer(tensor, device):
    """
    Safely transfers tensor to device with fallback to CPU if needed.

    Args:
        tensor (torch.Tensor): The tensor to transfer.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The transferred tensor.
    """
    try:
        return tensor.to(device)
    except Exception as e:
        if device.type == 'mps':
            print.warning(f"MPS transfer failed: {e}. Falling back to CPU")
            return tensor.cpu()
        raise

def safe_operation(operation, fallback_device='cpu'):
    """
    Decorator to safely handle operations that might not be supported on certain devices.

    Args:
        operation (callable): The operation to perform.
        fallback_device (str): Device to fall back to if operation fails.

    Returns:
        callable: Wrapped function that handles device fallback.
    """
    def wrapper(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            # Check if any args are tensors on MPS
            has_mps = any(isinstance(arg, torch.Tensor) and arg.device.type == 'mps' 
                         for arg in args)
            if has_mps:
                # Move everything to fallback device
                cpu_args = [arg.to(fallback_device) if isinstance(arg, torch.Tensor) 
                          else arg for arg in args]
                cpu_kwargs = {k: v.to(fallback_device) if isinstance(v, torch.Tensor)
                            else v for k, v in kwargs.items()}
                # Perform operation on CPU
                result = operation(*cpu_args, **cpu_kwargs)
                # Move back to MPS if possible
                try:
                    return safe_device_transfer(result, 'mps')
                except:
                    return result
            raise
    return wrapper

@safe_operation
def var_mean_safe(tensor, *args, **kwargs):
    """
    Safely computes the variance and mean of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        *args: Additional positional arguments for torch.var_mean.
        **kwargs: Additional keyword arguments for torch.var_mean.

    Returns:
        tuple: Variance and mean of the tensor.
    """
    return torch.var_mean(tensor, *args, **kwargs)

def safe_model_forward(model, input_tensor, conditions_tensor, device):
    """
    Safely performs the model's forward pass with proper device handling.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): The input tensor.
        conditions_tensor (torch.Tensor): The conditions tensor.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The model's output.
    """
    # Ensure inputs are on the correct device
    input_tensor = safe_device_transfer(input_tensor, device)
    conditions_tensor = safe_device_transfer(conditions_tensor, device)

    try:
        # Try normal forward pass
        output = model(input_tensor, conditions_tensor)
    except Exception as e:
        if device.type == 'mps':
            print.warning(f"MPS forward pass failed: {e}. Attempting CPU fallback")
            # Move model and inputs to CPU
            cpu_model = model.cpu()
            cpu_input = input_tensor.cpu()
            cpu_conditions = conditions_tensor.cpu()

            # Run on CPU
            output = cpu_model(cpu_input, cpu_conditions)

            # Move output back to MPS if possible
            try:
                output = safe_device_transfer(output, device)
            except:
                print.warning("Could not move output back to MPS, keeping on CPU")

            # Move model back to original device
            model.to(device)
        else:
            raise

    return output

def log_exception(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook to log uncaught exceptions.

    Args:
        exc_type (type): Exception type.
        exc_value (Exception): Exception value.
        exc_traceback (traceback): Traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Call the default exception handler for KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))



from utils import get_n_instruments
from models.build_model import build_model
from data.data_processing_reverse import ind_tensor_to_mid, ind_tensor_to_str

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.

    Args:
        lst (list): The list to divide.
        n (int): The chunk size.

    Yields:
        list: Chunks of the original list.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def avoid_overly_dense_notes_in_midi(mid, max_density=5, window_size=0.5):
    """
    Limits the number of notes that start within any sliding window of size window_size.
    Keeps the top max_density notes based on velocity within each window.

    Args:
        mid (pretty_midi.PrettyMIDI): The MIDI object to process.
        max_density (int): Maximum number of notes allowed per window.
        window_size (float): Size of the time window in seconds.
    """
    for instrument in mid.instruments:
        # Sort notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda x: x.start)
        # Sort notes by velocity descending
        sorted_by_velocity = sorted(sorted_notes, key=lambda x: x.velocity, reverse=True)

        kept_notes = []
        start_times = []

        for note in sorted_by_velocity:
            left = note.start - window_size
            right = note.start + window_size
            # Count how many kept_notes have start times within [left, right]
            idx_left = bisect.bisect_left(start_times, left)
            idx_right = bisect.bisect_right(start_times, right)
            count = idx_right - idx_left
            if count < max_density:
                kept_notes.append(note)
                bisect.insort(start_times, note.start)

        # Sort kept_notes by start time
        kept_notes_sorted = sorted(kept_notes, key=lambda x: x.start)
        # Replace instrument's notes with kept_notes_sorted
        instrument.notes = kept_notes_sorted

def group_notes_for_chords_in_midi(mid, threshold=0.05):
    """
    Groups notes that start within threshold seconds of each other into chord groups.
    Logs the chord groups for each instrument.

    Args:
        mid (pretty_midi.PrettyMIDI): The MIDI object to process.
        threshold (float): Time threshold in seconds to consider notes as part of the same chord.
    """
    for instrument in mid.instruments:
        sorted_notes = sorted(instrument.notes, key=lambda x: x.start)
        chord_groups = []
        current_group = []
        last_start = None

        for note in sorted_notes:
            if last_start is None or abs(note.start - last_start) <= threshold:
                current_group.append(note)
            else:
                if current_group:
                    chord_groups.append(current_group)
                current_group = [note]
            last_start = note.start

        if current_group:
            chord_groups.append(current_group)

        # Log the chord groups
        print(f"Instrument {instrument.program}: {len(chord_groups)} chord groups identified.")
        for idx, group in enumerate(chord_groups, 1):
            note_names = [pretty_midi.note_number_to_name(n.pitch) for n in group]
            print(f"  Chord {idx}: {', '.join(note_names)} at {group[0].start:.2f}s")

def generate(
    model,
    maps,
    device,
    out_dir,
    conditioning,
    mode,                        # 'collab' or 'full'
    short_filename=False,
    penalty_coeff=0.5,
    discrete_conditions=None,
    continuous_conditions=None,  # List of [valence, arousal_val] pairs
    max_input_len=1024,
    amp=True,
    step=None, 
    gen_len=2048,
    temperatures=[1.2, 1.2],
    top_k=-1, 
    top_p=0.7,
    debug=False,
    varying_condition=None,
    seed=-1,
    verbose=False,
    primers=[["<START>"]],
    min_n_instruments=2,
    max_density=10,
    window_size=0.5,
    chord_threshold=0.05
):
    """
    Generates symbolic music with improved MPS device handling and safety mechanisms.

    Args:
        model (torch.nn.Module): The PyTorch model for generation.
        maps (dict): Mapping dictionaries for tokens and events.
        device (torch.device): The device to run the model on.
        out_dir (str): Output directory for MIDI files.
        conditioning (str): Type of conditioning ('none', 'discrete_token', etc.).
        mode (str): Mode of generation ('collab' or 'full').
        short_filename (bool): Whether to use short filenames for outputs.
        penalty_coeff (float): Coefficient for penalizing repeating notes.
        discrete_conditions (torch.Tensor, optional): Discrete conditioning tokens.
        continuous_conditions (torch.Tensor, optional): Continuous conditioning values.
        max_input_len (int): Maximum input length in tokens.
        amp (bool): Whether to use automatic mixed precision.
        step (str, optional): Step identifier for filenames.
        gen_len (int): Maximum generation length in tokens.
        temperatures (list): Temperature values for sampling.
        top_k (int): Top-k sampling parameter.
        top_p (float): Top-p (nucleus) sampling parameter.
        debug (bool): Whether to run in debug mode (don't save files).
        varying_condition (list, optional): Varying conditions per token.
        seed (int): Random seed.
        verbose (bool): Whether to enable verbose print.
        primers (list): List of primer token lists.
        min_n_instruments (int): Minimum number of instruments required to save MIDI.
        max_density (int): Maximum number of notes allowed per window.
        window_size (float): Time window size in seconds for note density.
        chord_threshold (float): Time threshold in seconds to group notes into chords.

    Returns:
        tuple: (redo_primers, redo_discrete_conditions, redo_continuous_conditions)
    """
    @safe_operation
    def optimized_token_generation(
    model,
    input_tensor,
    conditions_tensor,
    maps,
    device,
    temp_tensor,
    top_k=-1,
    top_p=0.7,
    penalty_coeff=0.5,
    repeat_counts=None,
    exclude_symbols=None,
    batch_size=8  # Added batch size parameter
):
        """
        Optimized token generation function with MPS-specific improvements.
        """
        # Ensure conditions_tensor has the correct shape
        conditions_tensor = conditions_tensor.to(device)
        conditions_tensor = conditions_tensor.view(batch_size, -1)

        # Cache frequently used tensors on device
        cached_model = model.to(device)
        cached_conditions = conditions_tensor.to(device)
        cached_temp = temp_tensor.to(device)

        # Memory management for MPS
        if device.type == 'mps':
            torch.mps.empty_cache()

        # Batch processing setup
        input_chunks = torch.split(input_tensor, batch_size, dim=1)
        outputs = []

        # Process in batches with optimized device handling
        for chunk in input_chunks:
            with torch.no_grad():
                if device.type == 'mps':
                    torch.mps.synchronize()

                # Ensure chunk has the correct dimensions
                chunk = chunk.to(device)
                chunk_output = cached_model(chunk.t(), cached_conditions)
                chunk_output = chunk_output.permute(1, 0, 2)

                outputs.append(chunk_output)

        # Combine batch results
        # Ensure all tensors in outputs have the same shape except in dimension 1
        expected_shape = outputs[0].shape
        for out_tensor in outputs[1:]:
            assert out_tensor.shape == expected_shape, f"Tensor shape mismatch: {out_tensor.shape} vs {expected_shape}"

        output = torch.cat(outputs, dim=1)
        output = output[-1, :, :]  # Get final timestep

        # Safe handling of numerical operations
        output = safe_mps_operation(torch.nan_to_num, device, output, 0.0)

        # Exclude symbols with optimized indexing
        if exclude_symbols:
            exclude_indices = torch.tensor(
                [maps["tuple2idx"][symbol] for symbol in exclude_symbols if symbol in maps["tuple2idx"]],
                device=output.device
            )
            output.index_fill_(1, exclude_indices, float('-inf'))

        # Apply repeat penalty efficiently
        if penalty_coeff > 0 and repeat_counts is not None:
            repeat_counts_tensor = torch.tensor(repeat_counts, device=output.device)
            temp_multiplier = safe_mps_operation(
                lambda tensor: torch.maximum(
                    torch.zeros_like(tensor),
                    torch.log((tensor + 1) / 4) * penalty_coeff
                ), device,
                repeat_counts_tensor
            )
            cached_temp = cached_temp + temp_multiplier

        # Temperature scaling with safe handling
        output = output / cached_temp.t()

        # Optimized top-k sampling
        if top_k > 0 and top_k < output.size(-1):
            output, top_inds = safe_mps_operation(torch.topk, device, output, top_k, dim=-1)
        else:
            output, top_inds = safe_mps_operation(torch.topk, device, output,  output.size(-1), dim=-1)

        # Optimized top-p sampling
        if 0 < top_p < 1:
            probs = safe_mps_operation(F.softmax, device, output, -1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Efficient masking
            mask = cumulative_probs > top_p
            mask[:, 0] = False
            sorted_probs[mask] = 0

            # Restore original order
            probs.scatter_(1, sorted_indices, sorted_probs)
            output = torch.log(probs + 1e-8)  # Add small epsilon for numerical stability

        # Final sampling with safety checks
        output = safe_mps_operation(F.softmax, device, output, -1)

        try:
            sampled_inds = safe_mps_operation(
                lambda x: torch.multinomial(x, 1, replacement=True), device,
                output
            )
        except RuntimeError:  # Handle potential sampling issues
            # Fallback to argmax if sampling fails
            sampled_inds = output.argmax(dim=-1, keepdim=True)

        final_inds = safe_mps_operation(
            lambda x, y: x.gather(1, y), device, 
            top_inds,
            sampled_inds,
        ).t()

        # Update repeat counts efficiently
        if repeat_counts is not None:
            num_choices = torch.sum(output > 0, dim=-1)
            repeat_counts = [
                count + 1 if choices <= 2 else count // 2
                for count, choices in zip(repeat_counts, num_choices)
            ]

        # Final memory cleanup
        if device.type == 'mps':
            torch.mps.empty_cache()

        return final_inds, repeat_counts
    if device.type == 'mps':
        torch.mps.empty_cache()
    if not debug:
        os.makedirs(out_dir, exist_ok=True)

    model = model.to(device)
    model.eval()

    # Pruning and Quantization
    model = prune_model_weights(model, amount=0.2)
    #model = quantize_model(model, device.type)

    # Safe model compilation
    model = compile_model_safely(model, device)


    assert len(temperatures) in (1, 2), "Temperatures list must have length 1 or 2."

    # Prepare conditions
    if varying_condition is not None:
        varying_condition = [safe_device_transfer(vc, device) for vc in varying_condition]
        batch_size = varying_condition[0].size(0)
    else:
        try:
            continuous_conditions = safe_device_transfer(
                torch.FloatTensor(continuous_conditions), device
            ) if continuous_conditions is not None else None
        except:
            continuous_conditions = None

        if conditioning == "none":
            batch_size = len(primers)
        elif conditioning == "discrete_token":
            assert discrete_conditions is not None
            discrete_conditions_tensor = []
            for condition_sample in discrete_conditions:
                discrete_indices = [maps["tuple2idx"][symbol] for symbol in condition_sample]
                discrete_conditions_tensor.append(discrete_indices)
            discrete_conditions_tensor = safe_device_transfer(
                torch.LongTensor(discrete_conditions_tensor).t(), device
            )
            batch_size = discrete_conditions_tensor.size(1)
        elif conditioning in ("continuous_token", "continuous_concat"):
            batch_size = len(continuous_conditions)
        else:
            batch_size = 1

    # Initialize repeat counts
    repeat_counts = [0 for _ in range(batch_size)]

    # Prepare exclude symbols
    exclude_symbols = [
        symbol for symbol in maps["tuple2idx"].keys()
        if (isinstance(symbol, tuple) and len(symbol) == 1 and isinstance(symbol[0], str) and symbol[0].startswith("<"))
        or (isinstance(symbol, str) and symbol.startswith("<"))
    ]

    # Initialize generation tensor
    gen_song_tensor = safe_device_transfer(torch.LongTensor([]), device)

    # Convert primers to indices
    if not isinstance(primers, list):
        primers = [[primers]]
    primer_inds = [[maps["tuple2idx"][sym] for sym in p] for p in primers]

    gen_inds = torch.LongTensor(primer_inds)
    null_conditions_tensor = safe_device_transfer(torch.FloatTensor([np.nan, np.nan]), device)

    if len(primers) == 1:
        gen_inds = gen_inds.repeat(batch_size, 1)
        null_conditions_tensor = null_conditions_tensor.repeat(batch_size, 1)

    # Set up conditions
    if conditioning == "continuous_token":
        max_input_len -= 2
        conditions_tensor = continuous_conditions
    elif conditioning == "continuous_concat":
        conditions_tensor = continuous_conditions
    elif conditioning == "discrete_token":
        max_input_len -= discrete_conditions_tensor.size(0)
        conditions_tensor = null_conditions_tensor
    else:
        conditions_tensor = null_conditions_tensor

    # Handle varying conditions
    if varying_condition is not None:
        varying_condition = [vc.to(device) for vc in varying_condition]

    gen_inds = safe_device_transfer(gen_inds.t(), device)

    # Generation loop
    with torch.no_grad():
        if amp and device.type == 'mps':
            autocast = torch.amp.autocast('mps', enabled=True)
        elif amp and device.type == 'cuda':
            autocast = torch.amp.autocast("cuda", enabled=True)
        else:
            autocast = torch.amp.autocast("cpu", enabled=False)

        with autocast:
            for i in range(1, gen_len + 1):
                if verbose and i % 100 == 0:
                    print(f"Generating token {i}/{gen_len}")

                # Append new tokens
                gen_song_tensor = torch.cat((gen_song_tensor, gen_inds), 0)

                # Prepare model input
                input_ = gen_song_tensor
                if len(gen_song_tensor) > max_input_len:
                    input_ = gen_song_tensor[-max_input_len:, :]

                if conditioning == "discrete_token":
                    input_ = torch.cat((discrete_conditions_tensor, input_), 0)

                # Handle varying conditions
                if varying_condition is not None:
                    valences = varying_condition[0][:, i-1]
                    arousals = varying_condition[1][:, i-1]
                    conditions_tensor = torch.cat([valences[:, None], arousals[:, None]], dim=-1)

                # Calculate effective temperatures
                effective_temps = []
                for j in range(batch_size):
                    gen_idx = gen_inds[0, j].item()
                    gen_tuple = maps["idx2tuple"][gen_idx]
                    effective_temp = temperatures[1]
                    if isinstance(gen_tuple, tuple):
                        gen_event = maps["idx2event"][gen_tuple[0]]
                        if "TIMESHIFT" in gen_event:
                            effective_temp = temperatures[0]
                    effective_temps.append(effective_temp)

                temp_tensor = safe_device_transfer(torch.Tensor([effective_temps]), device)

                # Use optimized token generation
                gen_inds, repeat_counts = optimized_token_generation(
                    model=model,
                    input_tensor=input_,
                    batch_size=batch_size,
                    conditions_tensor=conditions_tensor,
                    maps=maps,
                    device=device,
                    temp_tensor=temp_tensor,
                    top_k=top_k,
                    top_p=top_p,
                    penalty_coeff=penalty_coeff,
                    repeat_counts=repeat_counts,
                    exclude_symbols=exclude_symbols
                )

    # Rest of the function (MIDI generation and post-processing)
    redo_primers, redo_discrete_conditions, redo_continuous_conditions = [], [], []

    for idx in range(gen_song_tensor.size(-1)):
        # Generate filename
        if short_filename:
            out_file_path = f"{idx}"
        else:
            if step is None:
                now = datetime.datetime.now()
                out_file_path = now.strftime("%Y_%m_%d_%H_%M_%S")
            else:
                out_file_path = step
            out_file_path += f"_{idx}"

        if seed > 0:
            out_file_path += f"_s{seed}"

        if continuous_conditions is not None and idx < continuous_conditions.size(0):
            cond_vals = continuous_conditions[idx, :].tolist()
            cond_str = [str(round(c, 2)).replace(".", "") for c in cond_vals]
            out_file_path += f"_V{cond_str[0]}_A{cond_str[1]}"

        out_file_path += ".mid"
        out_path_mid = os.path.join(out_dir, out_file_path)

        symbols = ind_tensor_to_str(gen_song_tensor[:, idx], maps["idx2tuple"], maps["idx2event"])
        n_instruments = get_n_instruments(symbols)

        if verbose:
            print("")

        if n_instruments >= min_n_instruments:
            mid = ind_tensor_to_mid(gen_song_tensor[:, idx], maps["idx2tuple"], maps["idx2event"], verbose=False)
            #avoid_overly_dense_notes_in_midi(mid, max_density=max_density, window_size=window_size)
            #group_notes_for_chords_in_midi(mid, threshold=chord_threshold)

            if not debug:
                mid.write(out_path_mid)
                if verbose:
                    print(f"Saved to {out_path_mid}")
        else:
            print(f"Only has {n_instruments} instruments, not saving.")
            if conditioning == "none":
                redo_primers.append(primers[idx])
                redo_discrete_conditions = None
                redo_continuous_conditions = None
            elif conditioning == "discrete_token":
                redo_discrete_conditions.append(discrete_conditions[idx])
                redo_continuous_conditions = None
                redo_primers = primers
            else:
                redo_discrete_conditions = None
                cvals = continuous_conditions[idx, :].tolist()
                redo_continuous_conditions.append(cvals)
                redo_primers = primers

    return redo_primers, redo_discrete_conditions, redo_continuous_conditions

def generate_midi(
    model_dir: str,
    cpu: bool,
    num_runs: int,
    gen_len: int,
    max_input_len: int,
    temp: List[float],
    topk: int,
    topp: float,
    debug: bool,
    seed: int,
    no_amp: bool,
    conditioning: str,
    penalty_coeff: float,
    quiet: bool,
    short_filename: bool,
    batch_size: int,
    min_n_instruments: int,
    batch_gen_dir: str,
    out_dir: str,
    arousal_feature: str,
    valence: List[Optional[float]],
    arousal_val: List[Optional[float]],
    max_density: int,
    window_size: float,
    chord_threshold: float,
    prune: int
):
    """
    Main function to generate MIDI files based on the provided parameters.

    Args:
        All parameters correspond to the command-line arguments.
    """

    # If conditioning == "none", valence/arousal_val must be [None]
    # Get the parameters of the function
    import inspect
    parameters = inspect.signature(generate_midi).parameters

    print(f"cpu: {cpu}")

    
    if conditioning == "none":
        if not (valence == [None] and arousal_val == [None]):
            raise ValueError("If conditioning == 'none', do not specify valence/arousal_val.")
    else:
        # If conditioning is used, we expect real values
        if valence == [None] or arousal_val == [None]:
            raise ValueError("If conditioning is used, specify valence and arousal_val explicitly.")
    main_output_dir = out_dir
    validate_output_dir(main_output_dir)
    print(f"Model directory: {model_dir}")
    model_root_dir = os.path.abspath(os.path.join(main_output_dir, 'models', model_dir))
    print(f"model_root_dir: {model_root_dir}")
    # Build final output directory
    if not os.path.exists(model_root_dir):
        raise ValueError(f"Model directory does not exist: {model_root_dir}")
    midi_output_dir = os.path.join(out_dir)
    if midi_output_dir is not None and midi_output_dir != "" and batch_gen_dir is not None and batch_gen_dir != "":
        midi_output_dir = os.path.join(midi_output_dir, batch_gen_dir)
    if midi_output_dir == "" or midi_output_dir is None:
        raise ValueError("Output directory is empty.")
    if not debug:
        os.makedirs(midi_output_dir, exist_ok=True)

    # Optimization 5: Load model files efficiently
    model_fp = os.path.join(model_root_dir, 'model.pt')
    mappings_fp = os.path.join(model_root_dir, 'mappings.pt')
    config_fp = os.path.join(model_root_dir, 'model_config.pt')

    if not os.path.exists(mappings_fp):
        raise ValueError(f"Mapping file not found: {mappings_fp}")

    if not os.path.exists(config_fp):
        raise ValueError(f"model_config.pt file not found: {config_fp}")

    device = None
    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else
                              "cpu")
    config = torch.load(config_fp, map_location=device)
    maps = torch.load(mappings_fp, map_location=device)

    # Optimization 6: Build model and move to device once
    model, _ = build_model(None, load_config_dict=config)
    verbose = not quiet

    if verbose:
        if device.type == 'mps':
            print("Using MPS (Apple Silicon) backend for PyTorch.")
        elif device.type == 'cuda':
            print("Using CUDA (GPU) backend for PyTorch.")
        else:
            print("Using CPU backend for PyTorch.")

    model = model.to(device)
    # Optimization 7: Load model weights efficiently
    if os.path.exists(model_fp):
        model.load_state_dict(torch.load(model_fp, map_location=device))
    elif os.path.exists(model_fp.replace("best_", "")):
        model.load_state_dict(torch.load(model_fp.replace("best_", ""), map_location=device))
    else:
        raise ValueError(f"Model weights not found in {model_fp} or {model_fp.replace('best_', '')}")

    # Prepare continuous conditions
    # e.g., if the user passed --valence 0.8 and --arousal_val 0.6
    # you get conditions = [[0.8, 0.6]]
    conditions = []
    if valence == [None]:
        # no conditioning or 'none'
        conditions = None
    elif len(valence) == 1:
        # repeat for batch_size
        for _ in range(batch_size):
            conditions.append([valence[0], arousal_val[0]])
    else:
        # e.g. multiple valence/arousal pairs
        for i in range(len(valence)):
            conditions.append([valence[i], arousal_val[i]])

    # Set up primers
    primers = [["<START>"]]
    discrete_conditions = None
    continuous_conditions = conditions

    # If user picks discrete_token:
    #   Convert val/ar into discrete tokens. We'll skip that here for brevity
    # If user picks none, we'll skip conditions
    # If user picks continuous_token or continuous_concat, we pass conditions

    # Actual run of the generation loop
    for run_idx in range(num_runs):
        primers_run = deepcopy(primers)
        discrete_conditions_run = deepcopy(discrete_conditions)
        continuous_conditions_run = deepcopy(continuous_conditions)

        # Keep calling generate() until all seeds are done
        # or until we run out of conditions
        while (primers_run != [] or discrete_conditions_run != [] or continuous_conditions_run != []):
            redo_primers, redo_discrete_conditions, redo_continuous_conditions = generate(
                model=model,
                maps=maps,
                device=device,
                out_dir=midi_output_dir,
                conditioning=conditioning,
                mode='full',
                short_filename=short_filename,
                penalty_coeff=penalty_coeff,
                discrete_conditions=discrete_conditions_run,
                continuous_conditions=continuous_conditions_run,
                max_input_len=max_input_len,
                amp=not no_amp,
                step=None,
                gen_len=gen_len,
                temperatures=temp,
                top_k=topk,
                top_p=topp,
                debug=debug,
                varying_condition=None,
                verbose=verbose,
                primers=primers_run,
                min_n_instruments=min_n_instruments,
                # Pass Task 6 and Task 9 parameters
                max_density=max_density,
                window_size=window_size,
                chord_threshold=chord_threshold
            )

            # Update for possible re-generation
            primers_run = redo_primers
            discrete_conditions_run = redo_discrete_conditions
            continuous_conditions_run = redo_continuous_conditions

    print("Generation completed.")

if __name__ == '__main__':
    # Set custom exception hook
    sys.excepthook = log_exception

    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_model_dir = os.path.abspath(os.path.join(script_dir, 'model'))
    code_utils_dir = os.path.join(code_model_dir, 'utils')
    sys.path.extend([code_model_dir, code_utils_dir])

    parser = ArgumentParser(description='Generates emotion-based symbolic music')

    # Restored / Merged arguments
    parser.add_argument('--model_dir', type=str, default='', help='Directory with model')
    parser.add_argument('--cpu', action='store_true',
                        help="Use CPU instead of GPU")
                        
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs')
    parser.add_argument('--gen_len', type=int, default=2048,
                        help='Max generation length in tokens')
    parser.add_argument('--max_input_len', type=int, default=1600,
                        help='Max input length in tokens')
    parser.add_argument('--temp', type=float, nargs='+', default=[1.2, 1.2],
                        help='Generation temperature (notes, rests)')
    parser.add_argument('--topk', type=int, default=-1,
                        help='Top-k sampling')
    parser.add_argument('--topp', type=float, default=0.7,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode (don't save files)")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('--no_amp', action='store_true',
                        help="Disable automatic mixed precision")
    parser.add_argument("--conditioning", type=str, default='continuous_concat',
                        choices=["none", "discrete_token", "continuous_token",
                                 "continuous_concat"],
                        help='Conditioning type')
    parser.add_argument('--penalty_coeff', type=float, default=0.5,
                        help="Coefficient for penalizing repeating notes")
    parser.add_argument("--quiet", action='store_true',
                        help="Suppress verbose output")
    parser.add_argument("--short_filename", action='store_true',
                        help="Use short filenames for output")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--min_n_instruments', type=int, default=1,
                        help='Minimum number of instruments required to save MIDI')
    parser.add_argument("--batch_gen_dir", type=str, default="",
                        help="Subdirectory name for batch generation")
    parser.add_argument("--out_dir", type=str, default="output",
                        help="Output directory for MIDI files")

    # Add back the original --arousal_feature argument
    parser.add_argument('--arousal_feature', type=str, default='note_density',
                        choices=['tempo', 'note_density'],
                        help='Feature to use as arousal feature in the model (if applicable)')

    # Renamed the continuous arousal argument to avoid conflict
    parser.add_argument('--valence', type=float, nargs='+', default=[0.8],
                        help='Continuous valence value(s) for conditioning')
    parser.add_argument('--arousal_val', type=float, nargs='+', default=[0.8],
                        help='Continuous arousal value(s) for conditioning')

    # Added arguments for Task 6 and Task 9
    parser.add_argument('--max_density', type=int, default=10,
                        help='Maximum number of notes allowed per window (Task 6)')
    parser.add_argument('--window_size', type=float, default=0.5,
                        help='Time window size in seconds for note density (Task 6)')
    parser.add_argument('--chord_threshold', type=float, default=0.05,
                        help='Time threshold in seconds to group notes into chords (Task 9)')
    parser.add_argument('--prune', type=int, default=-1,
                        help='Whether or not to prune the models weights (20%)')

    args = parser.parse_args()

    # Call the generate_midi function with parsed arguments
    generate_midi(
        model_dir=args.model_dir,
        cpu=args.cpu,
        num_runs=args.num_runs,
        gen_len=args.gen_len,
        max_input_len=args.max_input_len,
        temp=args.temp,
        topk=args.topk,
        topp=args.topp,
        debug=args.debug,
        seed=args.seed,
        no_amp=args.no_amp,
        conditioning=args.conditioning,
        penalty_coeff=args.penalty_coeff,
        quiet=args.quiet,
        short_filename=args.short_filename,
        batch_size=args.batch_size,
        min_n_instruments=args.min_n_instruments,
        batch_gen_dir=args.batch_gen_dir,
        out_dir=args.out_dir,
        arousal_feature=args.arousal_feature,
        valence=args.valence,
        arousal_val=args.arousal_val,
        max_density=args.max_density,
        window_size=args.window_size,
        chord_threshold=args.chord_threshold,
        prune=args.prune
    )