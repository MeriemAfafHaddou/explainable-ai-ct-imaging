import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def _save_gradients(self, module, grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, image, target_class):
        self.model.zero_grad()

        output = self.model(image)

        score = output[:, target_class]
        score.backward()

        activations = self.activations
        gradients = self.gradients

        # Average gradients over spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Keep only positive influence
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, output