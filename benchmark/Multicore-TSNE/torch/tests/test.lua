tsne = require 'tsne'

local tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
trainData = torch.load(train_file,'ascii').data:view(60000,32*32)
-- print(trainData)
Y = tsne(trainData:double(), n_components, perplexity, n_iter, angle, 24)
torch.save('mnist_embedding.t7', Y)