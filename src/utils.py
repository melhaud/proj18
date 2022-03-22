# store class indexes in a dictionary
from torchvision import transforms
from lightly.data import BaseCollateFunction
from torch import nn
import torch
from sklearn.preprocessing import normalize

def get_classes(dataset) -> dict:
    classes = {k:[] for k in dataset.class_to_idx}
    for i in range(len(dataset)):
        for cl in dataset.class_to_idx:
            if dataset[i][1] == dataset.class_to_idx[cl]:
                classes[cl].append(i)
    return classes


def generate_embeddings(model, dataloader):
    """
    Generates representations for all images in the dataloader with
    the given model
    """
    embeddings, filenames = [], []
    with torch.no_grad():
        for img, label, fnames in dataloader:
            img = img[0].to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings.cpu().numpy())
    return embeddings, filenames



# Standard augs
GaussianBlur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))

# Random augs
RandomHorizontalFlip = transforms.RandomHorizontalFlip(p=0.5)
RandomPerspective = transforms.RandomPerspective(distortion_scale=0.3, p=1.0)
# RandomGrayscale = transforms.RandomGrayscale(p=0.1) # not applicable if num channels < 3
RandomRotation = transforms.RandomRotation(30)
RandomAffine = transforms.RandomAffine(30)
RandomSolarize = transforms.RandomSolarize(threshold=192.0)
RandomAdjustSharpness = transforms.RandomAdjustSharpness(sharpness_factor=2)

# Sequential (каждый Sequential = 1 аугментация)
transformers_1 = nn.Sequential(GaussianBlur, RandomPerspective, RandomHorizontalFlip)
transformers_2 = nn.Sequential(RandomRotation, GaussianBlur, RandomSolarize)

# List of augmentations to be added
augs_list = [GaussianBlur, RandomHorizontalFlip, GaussianBlur, transformers_1, transformers_2]

# Final processing
# Attention! There is an unobvious choice of custom augmentations, but this is a limitation of the lightly itself.
augs_list.append(transforms.ToTensor())
transform = transforms.Compose(augs_list)
custom_collate_fn = BaseCollateFunction(transform)