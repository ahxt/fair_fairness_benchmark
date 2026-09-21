"""Run FFB DiffDP with the supplied ResNet-20 on the CelebA-A split."""

from reproduce_resnet18_celeba import main


if __name__ == "__main__":
    raise SystemExit(main(default_architecture="resnet20", default_method="diffdp"))
