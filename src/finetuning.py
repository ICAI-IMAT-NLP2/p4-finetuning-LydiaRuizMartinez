import torch
import torch.nn as nn
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model

class LoRA(nn.Module):
    def __init__(self, original_layer, r=4, alpha=32):
        """
        Low-Rank Adaptation (LoRA) module.
        
        Args:
            original_layer (nn.Module): The original layer to which LoRA is applied.
            r (int): Rank of the low-rank approximation.
            alpha (int): Scaling factor for the LoRA module.
        """
        super().__init__()
        assert isinstance(original_layer, nn.Linear), "LoRA only wraps nn.Linear layers."
        # TODO: Initialize LoRA parameters
        self.r = r
        self.alpha = alpha
        self.original_layer = original_layer

        # TODO: Low-rank matrices A and B for LoRA
        out_features = original_layer.out_features
        in_features = original_layer.in_features
        self.A = nn.Parameter(torch.empty(out_features, r))
        self.B = nn.Parameter(torch.zeros(r, in_features))

        # TODO: Initialize LoRA weights (B is zero-initialized, A is random)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
        # TODO: Scaling factor alpha 
        self.scaling = alpha / r

        # TODO: Freeze the original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        # TODO: Perform forward pass with low-rank update
        base = self.original_layer(x)
        delta = (x @ self.A) @ self.B
        return base + self.scaling * delta

def inject_lora_into_model(model, r=4, alpha=32, device='cpu'):
    """
    Inject LoRA layers into the linear layers of the attention modules of the model.
    
    Args:
        model (PreTrainedModel): The pre-trained model.
        r (int): Rank of the low-rank approximation.
        alpha (int): Scaling factor for LoRA.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    
    Returns:
        model (PreTrainedModel): The model with LoRA injected into attention layers.
    """
    # TODO: Iterate through all child modules of the model
    attention_name_hints = ("attention", "attn")
    stack = [(model, "")]  # (parent_module, path_string)

    while stack:
        parent_module, path = stack.pop()

        parent_class_lower = parent_module.__class__.__name__.lower()
        path_lower = path.lower()
        in_attention_context = any(h in parent_class_lower for h in attention_name_hints) or \
                            any(h in path_lower for h in attention_name_hints)


        # --- IMPORTANTE: el procesamiento debe estar DENTRO del bucle for ---
        for child_name, child_module in list(parent_module.named_children()):
            child_path = f"{path}.{child_name}" if path else child_name
        
            # TODO: Check if the child module is a linear layer of the attention module and create LoRA layer for linear module
            
            if isinstance(child_module, nn.Linear) and in_attention_context:
                wrapped = LoRA(child_module, r=r, alpha=alpha).to(device)
                setattr(parent_module, child_name, wrapped)
                # No descendemos dentro del hijo reemplazado
                continue

            # TODO: Recursively inject LoRA into child module
            stack.append((child_module, child_path))

    model.to(device)
    return model


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, model_hidden_size):
        """
        Creates trainable soft prompts to prepend to input embeddings.

        Args:
            prompt_length (int): Number of virtual tokens in the soft prompt.
            model_hidden_size (int): The hidden size of the pre-trained model.
        """
        super().__init__()
        # TODO: Initialize soft prompt embeddings
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(model_hidden_size)
        self.soft_prompt = nn.Parameter(torch.empty(self.prompt_length, self.hidden_size))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
        

    def forward(self, input_embeddings):
        """
        Forward pass to prepend soft prompts to input embeddings.

        Args:
            input_embeddings (torch.Tensor): The original input embeddings from the tokenizer.

        Returns:
            torch.Tensor: The concatenated soft prompts and original embeddings.
        """

        prompt = self.soft_prompt.to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        # TODO: Expand soft prompt to match batch size
        batch_size = input_embeddings.shape[0]
        prompt_expanded = prompt.unsqueeze(0).expand(batch_size, -1, -1)  # (B, P, H)

        # TODO: Concatenate soft prompt and input embeddings
        return torch.cat([prompt_expanded, input_embeddings], dim=1)
