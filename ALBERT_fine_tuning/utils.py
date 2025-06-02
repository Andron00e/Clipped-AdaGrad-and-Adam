import torchvision
import transformers

def get_per_layer_parameters(
        model, 
        independent_layer_types=None,
    ):
    """
    model: torch.nn.Module 
        pytorch model
    independent_layer_types: list
        list of modules which should be treated as a layer
    """

    if independent_layer_types is None:
        independent_layer_types = []

    # if it does not have any nested elements
    if len(list(model.children())) == 0:
        # if it has parameters, then return something useful
        if len(list(model.parameters())) != 0:
            return [{"params": model.parameters()}]
        # else return an empty list
        return []

    # if it has nested elements
    result = []
    for child_module in model.children():
        # if it is already an independent block (layer)
        if type(child_module) in independent_layer_types:
            result.append(
                {"params": child_module.parameters()} # an independent module
            )
        # otherwise get list of its parameters using recursion 
        else:
            result.extend(get_layers_parameters(child_module, independent_layer_types))
    return result

INDEP_LAYER_PER_NAME = {
    "resnet18-finetuning": [
        torchvision.models.resnet.BasicBlock,
    ],
    "bert": [
        transformers.models.albert.modeling_albert.AlbertEmbeddings,
        transformers.models.albert.modeling_albert.AlbertLayer
    ]
}
