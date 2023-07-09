import torchvision.transforms as tt
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import PIL
import io


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, f_input):
        self.loss = F.mse_loss(f_input, self.target)
        return f_input


def gram_matrix(inputs):
    a, b, c, d = inputs.size()
    features = inputs.view(a * b, c * d)
    gram = torch.mm(features, features.t())
    return gram.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, inputs):
        gram = gram_matrix(inputs)
        self.loss = F.mse_loss(gram, self.target)
        return inputs


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ImageLoader:
    def __init__(self, imsize=512):
        self.loader = tt.Compose([
            tt.Resize(imsize),
            tt.ToTensor()])
        self.unloader = tt.ToPILImage()

    def image_loader(self, image_name, device):
        image = Image.open(image_name)
        image_size = image.size
        image = PIL.ImageOps.pad(image, (512, 512))
        image = self.loader(image).unsqueeze(0)

        return image.to(device, torch.float), image_size

    def get_image(self, tensor, img_size):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        image = PIL.ImageOps.fit(image, img_size)
        bio = io.BytesIO()
        bio.name = 'output.jpeg'
        image.save(bio, 'JPEG')
        bio.seek(0)

        return bio


class NST:
    def __init__(self, device):
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.vgg19(pretrained=True).features.eval()
        torch.set_default_device(device)

    def get_style_model_and_losses(self, style_img, content_img):
        normalization = Normalization(self.cnn_normalization_mean,self.cnn_normalization_std)
        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)
        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers_default:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers_default:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    @staticmethod
    def get_input_optimizer(input_img):
        optimizer = torch.optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(self, content_img, style_img, input_img, num_steps=200,
                           style_weight=1000000, content_weight=1):
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)

        input_img.requires_grad_(True)
        model.eval()
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img


def run_nst(style_image, content_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_image, img_size = ImageLoader().image_loader(style_image, device)
    content_image, content_size = ImageLoader().image_loader(content_image, device)
    input_img = content_image.clone()
    outp = NST(device=device).run_style_transfer(content_image, style_image, input_img)
    output = ImageLoader().get_image(outp, content_size)
    return output
