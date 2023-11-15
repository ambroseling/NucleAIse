import torch
import torchvision
import numpy as np
#Class based hierarchical loss https://www.ijcai.org/proceedings/2021/0337.pdf
class HCLLoss:
    #Input to HCLLoss class:
    #   num_classes: number of classes we have in total
    #   edge_index: edge list that shows how the classes in your hierachy are connected
    #   pred_prob: logits from the last layer, assume normalized
    #   labels: the ground truth of the sample
    #    pred and labels have shape (num_samples or batch size,num_go_functions)
    #   thresh: a value controlling the amount of classes that contribute to the loss (needs to be experimented with)
    #Ouput of HCLLoss class:
    #   thresh_loss: the final loss value to be used for backprop 
    #   selected: the class indices that contributed to the loss

    def __init__(self,edge_index,num_classes,thresh):
        self.num_classes = num_classes
        self.base_loss = torch.nn.BCELoss(reduction="none")
        self.edge_index = edge_index
        #self.output_prob = output_prob #size (batch_size,num_go_functions)
        #self.target_prob = target_prob #size (batch_size,num_go_functions)
        self.thresh = thresh
    def get_parent_indices(self,index):
        edge_index = np.array(self.edge_index)
        mask = np.where(self.edge_index[0] == index)
        targets = edge_index[1][mask]
        return targets
    def compute_loss(self,output_prob,target_prob):
        ''' 
        implement HCL algorithm
        '''
        self.num_samples = output_prob.shape[0]
        initial_loss = torch.zeros(self.num_classes)
        selected = torch.zeros(self.num_classes)
        output_prob = torch.tensor(output_prob).float()
        target_prob = torch.tensor(target_prob).float()
        for c in range(self.num_classes):
            for n in range(self.num_samples):
                loss = self.base_loss(output_prob[n],target_prob[n]) # shape 1x num_go_functions
                #loss should have element wise loss
                print("Loss: ",loss)
                print("Current class: ",c)
                
                parents =  self.get_parent_indices(c)
                if len(parents) > 0:
                    print("Parent indices: ",parents)
                    parent_loss = loss[parents]
                    child_loss =  loss[c]
                    if torch.max(parent_loss)>child_loss:
                        subtracted_loss = torch.max(parent_loss)
                    else:
                        subtracted_loss = child_loss 
                else:
                   subtracted_loss = child_loss  
                initial_loss[c]+=subtracted_loss
                print("Initial Loss: ",initial_loss)
        sorted_loss ,sorted_indices = torch.sort(initial_loss,descending=False,stable=True)
        print("Sorted loss: ",sorted_loss)
        print("Sorted indices: ",sorted_indices)
        thresh_loss = 0
        thresh_classes = 0
        for c in range(self.num_classes):
            thresh_loss += sorted_loss[c]
            if thresh_loss + (c) > self.thresh +1:
                thresh_classes = c
                break
            else:
                continue
        print("Thresh classes: ",thresh_classes)
        for c in range(self.num_classes):
            if c<=thresh_classes:
                selected[sorted_indices[c]]=1
            else:
                selected[sorted_indices[c]]=0

        return thresh_loss,selected


                

#CHMLCN https://arxiv.org/pdf/2010.10151v1.pdf
class MCLoss():
    #Input to MCLoss:
    #   subclass matrix: similar to adjacency matrix, each row represents the classes j that are subclasses of class i
    #   pred_prob: logits from the last layer, assume normalized
    #   labels: the ground truth of the sample
    #Output of MCLoss:
    #   computes the loss and outputs it, used for backprop
    def __init__(self,device="cpu",subclass_matrix=None):
        ''' 
        '''
        self.device = device
        self.subclass_matrix = subclass_matrix
        self.loss_fn = torch.nn.BCELoss()
    def compute_loss(self,output,ground_truth):
        self.num_classes = output.shape[-1]
        M = self.subclass_matrix.bool()
        selected = torch.masked_select(torch.squeeze(output),M[0])
        hmcnn_out = torch.tensor([torch.max(torch.masked_select(torch.squeeze(output),M[i])) for i in range(self.num_classes)])
        H = ground_truth*hmcnn_out
        H_hat = H.repeat(self.num_classes,1)
        mcm,_ = torch.max(M*H,dim=1)
        l,_ = torch.max(M*H_hat,dim=1)
        output = (1-ground_truth)*mcm  + ground_truth*l
        loss = self.loss_fn(output.float(),ground_truth.float())
        return loss




def main():
    pred_prob = torch.tensor([[0.91,0.21,0.33,0.56,0.87,0.1]])
    labels = torch.tensor([[1,1,0,1,0,0]])
    edge_index = np.array([[1,3,0,2,2],[0,0,2,4,5]])
    num_classes = pred_prob.shape[-1]
    thresh = 2
    
    hcl = HCLLoss(edge_index=edge_index,num_classes=num_classes,thresh=thresh)
    thresh_loss,selected = hcl.compute_loss(pred_prob,labels)
    #thresh loss is the final loss value
    # can apply thresh.backward() here
    print("Thresh loss: ",thresh_loss)
    print("Selected nodes: ",selected)

    #HCL works
    subclass_matrix = torch.tensor([[0,1,0,1,0,0],
                       [1,0,0,0,0,0],
                       [1,1,0,1,0,0],
                       [0,0,1,0,0,0],
                       [1,1,1,1,1,0],
                       [1,1,1,1,0,1],])

    mcl = MCLoss("cpu", subclass_matrix)
    mcl_loss = mcl.compute_loss(pred_prob,labels)
    print("MCL Loss: ",mcl_loss)
    #MCLoss works

if __name__ == "__main__":
    main()
