transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

testset_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)
