"""Run FFB DiffDP with ResNet-18 on the same CelebA-A split as ERM."""

from reproduce_resnet18_celeba import main


if __name__ == "__main__":
    raise SystemExit(main(default_architecture="resnet18", default_method="diffdp"))
