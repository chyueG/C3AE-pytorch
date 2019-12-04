from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader import *
from tensorboardX import SummaryWriter
from datetime import datetime
import tqdm
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32,32,3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,1),

        )

        self.f1 = nn.Linear(32*4*4*3,12)
        #self.dropout = nn.Dropout2d(0.2)
        self.f2 = nn.Linear(12,1)




        """
        #version 1
        self.adp_pool = nn.AdaptiveAvgPool2d(1)
        self.f1  = nn.Conv2d(32,8,1)
        self.f2  = nn.Conv2d(8,1,1)
        
        
        version 2        
        self.conv1 = nn.Conv2d(3,32,3)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,3)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,3)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,32,3)
        self.bn4   = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,32,1)
        """

    def forward(self,input):
        """
        :param input: type list, should contain three scale face image
        :return: two full convolution features
        """
        input_0,input_1,input_2 = torch.split(input,3,dim=1)
        batch_size,h,w,c = input_0.shape
        """
        scale1 = self.__backbone_forward__(input_0)
        scale2 = self.__backbone_forward__(input_1)
        #scale3 = self.__backbone_forward__(input[2])
        all_fea = torch.cat([scale1,scale2],3)
        all_fea = all_fea.view(all_fea.size()[0],-1)
        in_channel = all_fea.size()[1]
        feat = self.__linear__(in_channel,8)(all_fea)
        pred = self.__linear__(8,1)(feat)

        """
        scale1  = self.feature(input_0)
        scale2  = self.feature(input_1)
        scale3  = self.feature(input_2)
        cat_scale = torch.cat([scale1,scale2,scale3],3)
        cat_scale = cat_scale.view(batch_size,-1)
        feat     = self.f1(cat_scale)
        #drop_fea = self.dropout(feat)
        #feat = self.adp_pool(feat_1)
        pred    = self.f2(feat)

        return feat,pred

    def __backbone_forward__(self,input):
        x = F.avg_pool2d(F.relu(self.bn1(self.conv1(input))),2,2)
        x = F.avg_pool2d(F.relu(self.bn2(self.conv2(x))),2,2)
        x = F.avg_pool2d(F.relu(self.bn2(self.conv3(x))),2,2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

    def __linear__(self,in_channel,out_channel):
        return nn.Linear(in_channel,out_channel).to(torch.device("cuda"))


def train_val(lambda_reg,model,device,data_loader,optimizer,scheduler,epoches):
    #get current work path
    pwd = os.path.dirname(os.path.abspath(__file__))
    if not pwd.endswith("/"):
        pwd = pwd + "/"


    save_dir = os.path.join(pwd , "logdir",datetime.now().strftime("%b%d_%H-%M-%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_dir = os.path.join(save_dir,"log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    summary_writer = SummaryWriter(log_dir)


    model_dir = os.path.join(save_dir,"model")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    #model.train()
    #scheduler.step()

    batch_len ={"train":len(data_loader["train"]),"val":len(data_loader["val"])}

    for epoch in range(epoches):
        for phase in ["train","val"]:
            total_loss = 0.0

            t_kl_loss = 0.0
            t_l1_loss = 0.0
            t_kr_loss = 0.0
            if phase == "train":
                #scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_data in tqdm.tqdm(data_loader[phase]):
                kl_reg = 0.0

                data,target,target_vec = batch_data["image"].to(device),batch_data["label"].to(device),batch_data["label_vec"].to(device)
                optimizer.zero_grad()
                if phase == "train":
                    feat,pred = model(data)
                else:
                    with torch.no_grad():
                        feat,pred = model(data)
                feat    = feat.view_as(target_vec)
                log_feat = F.log_softmax(feat)
                pred = pred.view_as(target)
                l1_loss   = F.l1_loss(pred,target)

                for param in model.f1.parameters():
                    kl_reg += torch.sum(torch.abs(param))

                kl_loss   = F.kl_div(log_feat,target_vec) + kl_reg*lambda_reg

                loss = l1_loss + kl_loss
                if phase == "train":
                    loss.backward()
                    optimizer.step()




                t_kl_loss += kl_loss
                t_kr_loss += kl_reg
                t_l1_loss += l1_loss
                total_loss += loss

            if phase == "train":
                model_name = "c3ae_" + str(epoch) + ".pth"
                torch.save({'epoch': epoch + 1, "state_dict": model.state_dict(), "opt_dict": optimizer.state_dict()},
                           model_dir + "/" + model_name)
                scheduler.step()
                print(" Train epoch :{} \t total loss is {:6f} kl loss is {:6f} l1 loss is {:6f}".format(epoch,total_loss.item()/batch_len[phase],\
                                                                                                      t_kl_loss.item()/batch_len[phase],\
                                                                                                      t_l1_loss.item()/batch_len[phase]))
                summary_writer.add_scalar("train l1 loss",l1_loss.item()/batch_len[phase],epoch)
                summary_writer.add_scalar("train kl regularize loss",t_kr_loss.item()/batch_len[phase],epoch)
                summary_writer.add_scalar("train kl loss",t_kl_loss.item()/batch_len[phase],epoch)
                summary_writer.add_scalar("train total loss",total_loss.item()/batch_len[phase],epoch)
            else:
                print(" Validation epoch :{} \t kl loss is {:6f} kr loss is {:6f} l1 loss is {:6f}".format(epoch,\
                                                                                                      t_kl_loss.item() / batch_len[phase], \
                                                                                                      t_kr_loss.item() /batch_len[phase], \
                                                                                                      t_l1_loss.item() / batch_len[phase]))
                summary_writer.add_scalar("val l1 loss", l1_loss.item() / batch_len[phase], epoch)
                #summary_writer.add_scalar("val kl regularize loss", t_kr_loss.item() / batch_len[phase], epoch)
                summary_writer.add_scalar("val kl loss", t_kl_loss.item() / batch_len[phase], epoch)
                summary_writer.add_scalar("val total loss", total_loss.item() / batch_len[phase], epoch)



    summary_writer.close()

def eval(model,device,test_loader):
    model.eval()
    test_loss = 0.0
    error_age = 0.0
    test_cnt = {}
    with torch.no_grad():
        for data in test_loader:
            cur_error = 0.0
            image,target,target_vec = data["image"].to(device),data["label"].to(device),data["label_vec"].to(device)
            feat,pred  = model(image)
            log_feat   = F.log_softmax(feat)
            pred = pred.view_as(target)
            test_loss += F.l1_loss(pred,target) + F.kl_div(feat,target_vec)
            error_age += torch.abs(pred - target)
            cur_error = torch.abs(pred-target)
            age_idx = torch.floor(target/10.0)
            if age_idx.item() in test_cnt:
                test_cnt[age_idx.item()].append(cur_error)
            else:
                test_cnt[age_idx.item()] = [cur_error]

            print("pred age is {} actual age is {}".format(pred.item(),target.item()))
            #pred    = output.argmax(dim=1,keepdim=True)
            #correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    mae        = error_age / len(test_loader.dataset)
    print("Test set:Average loss is {:.4f}, Mean average error  : ({:.0f}%)\n".format(test_loss,mae.item()))
    return test_cnt


def main(train_list,im_root,val_list,val_root,resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = FaceDataset(train_list,im_root,transform=transforms.Compose([Resize((64,64)),ToTensor()]))
    train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
    val_loader = DataLoader(FaceDataset(val_list,val_root,transform=transforms.Compose([Resize((64,64)),ToTensor()])),batch_size=32)
    dataLoader = {"train":train_loader,"val":val_loader}
    if resume:
        ckpt = torch.load("/c3ae_11.pth")

    net  = Backbone()
    net.load_state_dict(ckpt["state_dict"],strict=False)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    epoch = 30
    lambda_kl = 1e-5
    train_val(lambda_kl, net, device, dataLoader, optimizer, scheduler, epoch)

def test(test_list,im_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader =  DataLoader(FaceDataset(test_list,im_root,transform=transforms.Compose([Resize((64,64)),ToTensor()])),batch_size=1)
    net = Backbone()
    #net = net.to(device)
    checkpoint = torch.load("/c3ae_29.pth")
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)

    test_cnt = eval(net,device,test_loader)
    return test_cnt


if __name__=="__main__":
    train_list = "imdb_train.txt"
    im_root    = "/imdb-wike/imdb/"
    val_list   = "imdb_val.txt"
    val_root   = "/imdb-wike/imdb/"
    #main(train_list,im_root,val_list,val_root)
    test_cnt = test(val_list,val_root)
    for key in test_cnt.keys():
        mae = np.sum(test_cnt[key])*1.0/(len(test_cnt[key]))
        print("the range {} mae is {}".format(key,mae.item()))
