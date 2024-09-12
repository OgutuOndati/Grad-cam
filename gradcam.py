import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self.hook_layers()

    def hook_layers(self):
        # Hook for gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        # Hook for activations
        def forward_hook(module, input, output):
            self.activations.append(output)

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Convert gradients and activations to numpy
        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()

        # Calculate weights and CAM
        weights = torch.mean(torch.abs(self.gradients[0]), dim=0)
        cam = torch.sum(weights[None, :, None, None] * self.activations[0], dim=1).squeeze()
        
        # ReLU activation and normalization
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam
