import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import functional as TF


def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices

    returns: tensor shaped [m_1, m_2, m_3, m_4]

    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices

    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

def sample_within_bounds(signal, batch_index, y, x, channel_index):
    batch, height, width, channel = signal.shape
    #print(signal.shape)
    #print(batch_index.shape)
    x = torch.clamp(x, 0, width - 1)
    y = torch.clamp(y, 0, height - 1)
    batch_index = torch.clamp(batch_index, 0, batch - 1)
    channel_index = torch.clamp(channel_index, 0, channel - 1)

    index = torch.stack([torch.reshape(batch_index, [-1]), torch.reshape(y, [-1]), torch.reshape(x, [-1]),
                         torch.reshape(channel_index, [-1])], dim=1)
    #print(signal.shape)
    #print(index.shape)
    #result = torch.gather(signal, 0, index)
    result = gather_nd(signal,index)
    #print(result.shape)
    batch, height, width, channel = x.shape

    sample = torch.reshape(result, [batch, height, width, channel])

    return sample


def sample_bilinear(signal, batch_index, ry, rx, channel_index):
    signal_dim_y, signal_dim_x = signal.shape[1:-1]
    #print("signal_dim", signal_dim_y, signal_dim_x)

    # obtain four sample coordinates
    ix0 = rx.to(torch.int32)
    iy0 = ry.to(torch.int32)
    ix1 = torch.minimum(ix0 + 1, torch.tensor(signal_dim_x - 1))
    iy1 = torch.minimum(iy0 + 1, torch.tensor(signal_dim_y - 1))

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, batch_index, iy0, ix0, channel_index)
    signal_10 = sample_within_bounds(signal, batch_index, iy0, ix1, channel_index)
    signal_01 = sample_within_bounds(signal, batch_index, iy1, ix0, channel_index)
    signal_11 = sample_within_bounds(signal, batch_index, iy1, ix1, channel_index)

    ix1 = ix1.to(torch.float32)
    iy1 = iy1.to(torch.float32)
    # linear interpolation in x-direction
    fx1 = (ix1 - rx) * signal_00 + (1 - ix1 + rx) * signal_10
    fx2 = (ix1 - rx) * signal_01 + (1 - ix1 + rx) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (1 - iy1 + ry) * fx2


def polar_transformer(signal, height, width, shift, max_shift=20):
    batch, S, _, channel = signal.shape

    b = torch.arange(0, batch)
    h = torch.arange(0, height)
    w = torch.arange(0, width)
    c = torch.arange(0, channel)

    bb, hh, ww, cc = torch.meshgrid(b, h, w, c)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    shift = shift.view(batch, 1, 1, 2)

    shift_x = shift[..., 0:1]
    shift_y = shift[..., 1:2]

    shift_x = shift_x.repeat(1, height, width, channel)
    shift_y = shift_y.repeat(1, height, width, channel)

    radius = S / 2. - max_shift

    x = (S / 2. + shift_x) - radius / height * (height - 1 - hh) * torch.sin(2 * np.pi * ww / width)
    y = (S / 2. + shift_y) + radius / height * (height - 1 - hh) * torch.cos(2 * np.pi * ww / width)

    return sample_bilinear(signal, bb, y, x, cc)

def meshgrid_ij(*args):
    grid = torch.meshgrid(*args)
    grid = list(reversed(grid))
    return grid

#Projecitve Transform Function
def geometry_projector(signal, height, width, shift):

    batch, S, _, channel = signal.shape
    #print(channel)
    #print(S)

    b = torch.arange(0, batch)
    h = torch.arange(0, height * 2)
    w = torch.arange(0, width)
    c = torch.arange(0, channel)

    bb, hh, ww, cc = torch.meshgrid(b, h, w, c)
    #print(bb.shape)
    #print(ww.shape)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    shift = shift.view(batch, 1, 1, 2)

    shift_x = shift[..., 0:1]
    shift_y = shift[..., 1:2]

    shift_x = shift_x.repeat(1, height * 2, width, channel)
    shift_y = shift_y.repeat(1, height * 2, width, channel)

    tanhh = torch.tan(hh * np.pi / (height * 2))
    grd_height = -2
    s = S / 50
    #Projection
    x_bottom = ((S / 2. + shift_x) - s * grd_height * tanhh * torch.sin(2 * np.pi * ww / width))[:, height:, :, :]
    y_bottom = ((S / 2. + shift_y) + s * grd_height * tanhh * torch.cos(2 * np.pi * ww / width))[:, height:, :, :]

    x_half = -1 * torch.ones([batch, height, width, channel])
    y_half = -1 * torch.ones([batch, height, width, channel])

    x = torch.cat([x_half, x_bottom], dim=1)
    y = torch.cat([y_half, y_bottom], dim=1)


    projected_signal = sample_bilinear(signal, bb, y, x, cc)

    return projected_signal[:, int(height / 2): -int(height / 2)]

def degeometry_projector(signal, height, width, shift):
    
    batch, H, W, channel = signal.shape
    x_half_top = -1 * torch.ones([batch, 128, 512, channel])

    x_half_bottom = -1 * torch.ones([batch, 64, 512, channel])

    signal = torch.cat([x_half_top, signal],dim=1)
    signal = torch.cat([signal,x_half_bottom],dim = 1)

    batch, H, W, channel = signal.shape

    b = torch.arange(0, batch)
    h = torch.arange(0, height)
    w = torch.arange(0, width)
    c = torch.arange(0, channel)

    bb, hh, ww, cc = torch.meshgrid(b, h, w, c)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    shift = shift.view(batch, 1, 1, 2)

    shift_x = shift[..., 0:1]
    shift_y = shift[..., 1:2]

    shift_x = shift_x.repeat(1, height, width, channel)
    shift_y = shift_y.repeat(1, height, width, channel)

    grd_height = -2
    a = (ww - (height / 2. + shift_y)) / (height / 50 * grd_height)
    b = - (hh - (height / 2. + shift_x)) / (height / 50 * grd_height)

    angle = torch.atan2(a, b)

    x = ((angle + (angle < 0) * (2 * np.pi)) * W / (2 * np.pi))
    y = (torch.atan(-torch.sqrt(torch.square(a) + torch.square(b))) + np.pi) * H / (np.pi)

    projected_signal = sample_bilinear(signal, bb, y, x, cc)

    return projected_signal

def degeometry_p(signal, height, width, shift):

    batch, H, W, channel = signal.shape

    b = torch.arange(0, batch)
    h = torch.arange(0, height)
    w = torch.arange(0, width)
    c = torch.arange(0, channel)

    bb, hh, ww, cc = torch.meshgrid(b, h, w, c)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    shift = shift.view(batch, 1, 1, 2)

    shift_x = shift[..., 0:1]
    shift_y = shift[..., 1:2]

    shift_x = shift_x.repeat(1, height, width, channel)
    shift_y = shift_y.repeat(1, height, width, channel)

    grd_height = -2

    x = ww
    #y = (torch.atan(256-hh) + np.pi) * H / (np.pi)
    #y = (- ((torch.atan((hh) * 1 / H)) * H / np.pi) + H)
    y = (- ((torch.atan((H - hh) * 5 / H)) * H / np.pi) + H)
    #y = hh

    projected_signal = sample_bilinear(signal, bb, y, x, cc)

    return projected_signal

def ground2polar_pytorch(signal, height, width):

    batch, H, W, channel = signal.shape

    signal = signal.permute(0, 3, 1, 2)

    h = torch.arange(0, height)
    w = torch.arange(0, width)

    hh, ww = torch.meshgrid(h, w)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    x = ww
    y = (- ((torch.atan((H - hh) * 5 / H)) * H / np.pi) + H) #origin 5

    x = (x / (W - 1)) * 2 - 1  # Normalize x coordinate to range [-1, 1]
    y = (y / (H - 1)) * 2 - 1  # Normalize y coordinate to range [-1, 1]
    grid = torch.stack((x, y), dim=-1)
    grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)  # Shape: (batch, height, width, 2)
    #print(grid.shape)
    projected_signal = F.grid_sample(signal, grid, align_corners=True)

    return projected_signal.permute(0, 2, 3, 1)


def aerial2polar_pytorch(signal, height, width, max_shift=20):
    batch, H, W, channel = signal.shape

    signal = signal.permute(0, 3, 1, 2)

    S = height * 2
    h = torch.arange(0, height)
    w = torch.arange(0, width)

    hh, ww = torch.meshgrid(h, w)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    radius = S / 2. - max_shift

    x = (S / 2.) - radius / height * (height - 1 - hh) * torch.sin(2 * np.pi * ww / width)
    y = (S / 2.) + radius / height * (height - 1 - hh) * torch.cos(2 * np.pi * ww / width)

    x = (x / (S - 1)) * 2 - 1  # Normalize x coordinate to range [-1, 1]
    y = (y / (S - 1)) * 2 - 1  # Normalize y coordinate to range [-1, 1]

    grid = torch.stack((x, y), dim=-1)
    grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)  # Shape: (batch, height, width, 2)

    projected_signal = F.grid_sample(signal, grid, align_corners=True)

    return projected_signal.permute(0, 2, 3, 1)

#polar2aerial
def depolar_transformer_pytorch(signal, height, width, shift, max_shift=0):
    batch, H, W, channel = signal.shape

    signal = signal.permute(0, 3, 1, 2)

    h = torch.arange(0, height)
    w = torch.arange(0, width)

    hh, ww = torch.meshgrid(h, w)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    shift = shift.view(batch, 1, 1, 2)

    radius = height / 2. - max_shift

    a = - (ww - (radius)) * H / radius
    b = (hh - (radius)) * H / radius

    angle = torch.atan2(a, b)

    x = ((angle + (angle < 0) * (2 * np.pi)) * W / (2 * np.pi))
    y = (radius - torch.sqrt(torch.square(a)+torch.square(b)))

    x = (x / (W - 1)) * 2 - 1  # Normalize x coordinate to range [-1, 1]
    y = (y / (H - 1)) * 2 - 1  # Normalize y coordinate to range [-1, 1]

    grid = torch.stack((x, y), dim=-1)
    grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)  # Shape: (batch, height, width, 2)
    #print(grid.shape)

    projected_signal = F.grid_sample(signal, grid, align_corners=True)

    return projected_signal.permute(0, 2, 3, 1)
#polar2ground
def polar2ground_pytorch(signal, height, width, shift):
    #ex-polar2project

    batch, H, W, channel = signal.shape

    signal = signal.permute(0, 3, 1, 2)

    h = torch.arange(0, height)
    w = torch.arange(0, width)

    hh, ww = torch.meshgrid(h, w)

    hh = hh.to(torch.float32)
    ww = ww.to(torch.float32)

    x = ww
    y = H - H / 5 * torch.tan((H-hh) / H * np.pi)

    x = (x / (W - 1)) * 2 - 1  # Normalize x coordinate to range [-1, 1]
    y = (y / (H - 1)) * 2 - 1  # Normalize y coordinate to range [-1, 1]

    grid = torch.stack((x, y), dim=-1)
    grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)  # Shape: (batch, height, width, 2)



    projected_signal = F.grid_sample(signal, grid, align_corners=True)

    return projected_signal.permute(0, 2, 3, 1)

if __name__ == '__main__':
    from PIL import Image

    img = Image.open('./groundpolar_1227.png')#.resize((256, 256))
    img = np.asarray(img).astype(np.float32)
    img = torch.from_numpy(img)

    batch = 2

    signal = torch.stack([img] * batch, dim=0)  # shape = [batch, 3, 256, 256]
    shift = np.zeros((batch, 2)).astype(np.float32) #

    shift_torch = torch.from_numpy(shift)

    #image = geometry_projector(signal, 128, 512, shift_torch)
    #images = image
    original_images = depolar_transformer_pytorch(signal,256,256,shift_torch)

    images = original_images.detach().numpy()

    image0 = Image.fromarray(images[0].astype(np.uint8))
    image0.save('groundpolar_1227.png')
    image1 = Image.fromarray(images[1].astype(np.uint8))
    image1.save('tan1_torch.png')

    a = 1
