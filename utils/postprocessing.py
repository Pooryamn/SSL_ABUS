import torch
import torch.nn.functional as F

def post_process(x, device):
    
    output = torch.empty_like(x, device=device)
    x = x.unsqueeze(0)

    kernel = torch.ones([1, 1, 3], dtype=torch.float32).to(device)

    for i in range(x.shape[1]):
        x[:,i,:,0] = (x[:,i,:,0] > 0.5) * 1.0
        erosion = (F.conv1d(x[:,i,:,0], kernel, padding=1) == 3) * 1.0
        opening = (F.conv1d(erosion, kernel, padding=1) > 0) * 1.0
        dilation = (F.conv1d(opening, kernel, padding=1) > 0) * 1.0

        TMP = ((F.conv1d(dilation, kernel, padding=1) == 3) * 1.0)

        output[i,:,0] = TMP
        output[i,:,1:] = x[:,i,:,1:]

        mass_location = []
        BBOX_3D = []
        Flag = False

        for j in range(output.shape[1]):
            if(output[i,j,0] == 1):
                if(Flag == False):
                    # start of new mass
                    Flag = True
                    start = j
                    mass_location.append(output[i,j,1:])
                else:
                    # continue of mass
                    Flag = True
                    mass_location.append(output[i,j,1:])
            elif(output[i,j,0] == 0):
                if(Flag == True):
                    #end of mass
                    Flag = False
                    end = j
                    mass_location = np.array(mass_location)
                    X = np.mean(mass_location[:,0])
                    Y = np.mean(mass_location[:,1])
                    W = np.mean(mass_location[:,2])
                    H = np.mean(mass_location[:,3])
                  
                    BBOX_3D.append([X, Y, start, W, H, end - start])
                  
                    mass_location = list(mass_location)
                    mass_location = []
                else:
                    # simple non mass
                    Flag = False


    return output, BBOX_3D
