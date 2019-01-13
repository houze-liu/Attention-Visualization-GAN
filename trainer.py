import time
import datetime
import os
from modules import Generator, Discriminator
from torchvision.utils import save_image
import torch.optim
import visdom


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class Trainer(object):
    def __init__(self, data_loader, config):
        self.dataloader = data_loader
        self.imsize = config.imsize
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.g_dim = config.g_dim
        self.d_dim = config.d_dim
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lambda_gp = config.lambda_gp

        self.z_dim = config.z_dim
        self.num_iters = config.total_step
        self.num_iters_decay = config.iter_start_decay
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.lr_update_step = config.lr_iter_decay
        self.lr_decay = config.lr_decay
        self.model_save_step = config.model_save_step
        self.resume_iters = config.resume_iter
        self.version = config.version

        self.device = torch.device('cuda:0')

        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_dir = os.path.join(config.model_save_path, self.version)
        self.build_model()

    def build_model(self):
        self.G = Generator(image_size=self.imsize, z_dim=self.z_dim, conv_dim=self.g_dim)
        self.D = Discriminator(conv_dim=self.d_dim)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def gradient_penalty(self, y, x):
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def train(self):
        loss = {}
        vis = visdom.Visdom()

        data_iter = iter(self.dataloader)
        g_lr = self.g_lr
        d_lr = self.d_lr
        fixed_z = torch.randn(self.batch_size, self.z_dim).cuda()

        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        print('start training')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            try:
                x_mb, _ = next(data_iter)
            except:
                data_iter = iter(self.dataloader)
                x_mb, _ = next(data_iter)
            x_mb = x_mb.cuda()
            z = torch.randn(x_mb.size(0), self.z_dim).cuda()
            # train the discriminator
            x_fake = self.G(z)
            d_real = self.D(x_mb)
            d_fake = self.D(x_fake)
            d_loss_real = - torch.mean(d_real)
            d_loss_fake = torch.mean(d_fake)
            alpha = torch.rand(x_mb.size(0), 1, 1, 1).to(self.device)
            # interpolate between real data and fake data
            x_hat = (alpha * x_mb.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src = self.D(x_hat)

            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp

            d_loss.backward()
            self.d_optimizer.step()
            self.reset_grad()

            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            # train generator
            d_fake = self.D(self.G(z))
            g_loss = - torch.mean(d_fake)

            g_loss.backward()
            self.g_optimizer.step()
            self.reset_grad()

            loss['G/loss'] = g_loss.item()
            if (i + 1) % self.log_step == 0:
                # visualize real and fake imgs
                vis.images((x_fake + 1) / 2, win='fake_imgs')
                vis.images((x_mb + 1) / 2, win='real_imgs')
                # print and visualize losses
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                opts = dict(title='Losses', width=13, height=10, legend=list(loss.keys()))
                vis.line(Y=[list(loss.values())], X=[np.ones(len(loss))*(i+1)], win='Losses', \
                         update='append', opts=opts)
                print(log)

            if (i + 1) % self.lr_update_step == 0 and (i + 1) > self.num_iters_decay:
                g_lr = self.g_lr * self.lr_decay
                d_lr = self.d_lr * self.lr_decay
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # Sample images
            if (i + 1) % self.sample_step == 0:
                fake_images = self.G(fixed_z)
                save_image(denorm(fake_images.data),
                os.path.join(self.sample_path, '{}_fake.png'.format(i + 1)))

            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

# should have a function to claculate loss;
# so in the main loop, just need to call differnt losses if needed ot switch
