
from .jmvae_nf import JMVAE_NF_CELEBA as VAE_jnf_celeba
from .jmvae_nf import JMVAE_NF_DCCA_MNIST_SVHN as VAE_jnf_mnist_svhn_dcca
from .jmvae_nf.mnist_svhn_fashion import MNIST_SVHN_FASHION as VAE_jnf_msf
from .mmvae import MNIST_SVHN as VAE_mnist_svhn
from .mmvae.celeba import celeba as VAE_mmvae_celeba
from .mmvae.mnist_svhn_fashion import MNIST_SVHN_FASHION as VAE_mmvae_msf
from .mvae.celeba import celeba as VAE_mvae_celeba
from .mvae.mnist_svhn import MNIST_SVHN as VAE_mvae_mnist_svhn
from .mvae.msf import MNIST_SVHN_FASHION as VAE_mvae_msf

__all__ = [ 'VAE_mnist_svhn',
            'VAE_jnf_mnist_svhn_dcca',
            'VAE_jnf_celeba',
            'VAE_mmvae_celeba',
            'VAE_mvae_mnist_svhn',
            'VAE_mvae_celeba',
            'VAE_jnf_msf',
            'VAE_mmvae_msf', 
            'VAE_mvae_msf'

            ]