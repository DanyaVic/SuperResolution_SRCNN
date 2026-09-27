"""Модели super-resolution: SRCNN (обучается в проекте) и RRDBNet (генератор ESRGAN / Real-ESRGAN)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# SRCNN  (Dong et al., "Image Super-Resolution Using Deep Convolutional Networks", 2014/2016)
# ----------------------------------------------------------------------------------------------
class SRCNN(nn.Module):
    """Трёхслойная свёрточная сеть f1-f2-f3 (по умолчанию 9-5-5, 64/32 фильтра).

    Вход  — канал Y изображения, предварительно увеличенного бикубической интерполяцией до целевого
    размера, диапазон [0, 1]. Выход — восстановленный канал Y того же размера.
      слой 1: извлечение и представление патчей      F1 = ReLU(W1 * Y + B1)
      слой 2: нелинейное отображение                 F2 = ReLU(W2 * F1 + B2)
      слой 3: реконструкция                          F  = W3 * F2 + B3
    residual=True добавляет глобальную skip-связь (сеть учит только недостающие высокие частоты).
    """

    def __init__(self, n1=64, n2=32, f1=9, f2=5, f3=5, channels=1, residual=False, init="kaiming"):
        super().__init__()
        self.config = dict(n1=n1, n2=n2, f1=f1, f2=f2, f3=f3, channels=channels, residual=residual)
        self.residual = residual
        # padding='same' + replicate: выход того же размера, без тёмной рамки по краям
        self.conv1 = nn.Conv2d(channels, n1, f1, padding=f1 // 2, padding_mode="replicate")
        self.conv2 = nn.Conv2d(n1, n2, f2, padding=f2 // 2, padding_mode="replicate")
        self.conv3 = nn.Conv2d(n2, channels, f3, padding=f3 // 2, padding_mode="replicate")
        self.init_weights(init)

    def init_weights(self, mode="kaiming"):
        """'paper' — N(0, 0.001) как в статье (медленный старт при SGD);
        'kaiming' — He-инициализация скрытых слоёв, заметно быстрее сходится с Adam."""
        for m in (self.conv1, self.conv2):
            if mode == "paper":
                nn.init.normal_(m.weight, 0.0, 0.001)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.conv3.weight, 0.0, 0.001)
        nn.init.zeros_(self.conv3.bias)

    def last_layer_params(self):
        return list(self.conv3.parameters())

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.conv3(y)
        return x + y if self.residual else y


# ----------------------------------------------------------------------------------------------
# RRDBNet — генератор ESRGAN (Wang et al., 2018) и Real-ESRGAN (Wang et al., 2021)
# Совместим с официальными весами xinntao/Real-ESRGAN и BasicSR.
# ----------------------------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    """5 свёрток с плотными связями: каждая получает конкатенацию всех предыдущих карт."""

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # residual scaling β = 0.2


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 RDB внутри ещё одной residual-связи (без BatchNorm)."""

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Генератор ESRGAN: conv → 23×RRDB → conv (+skip) → 2× (nearest-upsample ×2 + conv) → conv → conv.

    Для scale=2 (Real-ESRGAN x2plus) вход сначала сжимается pixel-unshuffle ×2 (3→12 каналов),
    поэтому структура тела та же, а выход всё равно ×4 от «сжатого» входа = ×2 от исходного.
    """

    def __init__(self, in_ch=3, out_ch=3, scale=4, nf=64, nb=23, gc=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            in_ch *= 4
        elif scale == 1:
            in_ch *= 16
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            x = F.pixel_unshuffle(x, 4)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
