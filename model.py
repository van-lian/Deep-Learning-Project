import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths
ENCODER_PATH = 'C:/Deep Learning/Projects/Code/Image Caption Generation/Model/encoder-5-3000.pkl'
DECODER_PATH = 'C:/Deep Learning/Projects/Code/Image Caption Generation/Model/decoder-5-3000.pkl'
VOCAB_PATH = 'C:/Deep Learning/Projects/Code/Image Caption Generation/vocab.pkl'

# Constants for the architecture
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generate captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    def beam_search(self, features, beam_width=5):
        """Generate captions using beam search"""
        # Initialize beam
        start_token = self.vocab('<start>')
        beams = [([start_token], 0.0, None)]  # (tokens, score, states)
        
        for _ in range(self.max_seq_length):
            new_beams = []
            for tokens, score, states in beams:
                if tokens[-1] == self.vocab('<end>'):
                    new_beams.append((tokens, score, states))
                    continue
                    
                # Prepare inputs
                inputs = torch.LongTensor([tokens[-1]]).to(device)
                inputs = self.embed(inputs).unsqueeze(1)
                
                # Forward pass
                hiddens, states = self.lstm(inputs, states)
                outputs = self.linear(hiddens.squeeze(1))
                log_probs = F.log_softmax(outputs, dim=1)
                
                # Get top k candidates
                topk_scores, topk_tokens = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    new_tokens = tokens + [topk_tokens[0,i].item()]
                    new_score = score + topk_scores[0,i].item()
                    new_beams.append((new_tokens, new_score, states))
            
            # Keep top beam_width candidates
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        return beams[0][0]  # Return best sequence

def load_image(image_path, transform=None):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
        
    return image

def PretrainedResNet(image_path, encoder_path=ENCODER_PATH,
                    decoder_path=DECODER_PATH,
                    vocab_path=VOCAB_PATH,
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS):
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build models
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size,
                        len(vocab), num_layers)
    
    # Load the trained model parameters with proper device mapping
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    # Set to eval mode and move to device
    encoder = encoder.eval().to(device)
    decoder = decoder.eval().to(device)
    
    # Prepare an image
    image = load_image(image_path, transform)
    image_tensor = image.to(device)
    
    # Generate a caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    
    # (1, max_seq_length) -> (max_seq_length)
    sampled_ids = sampled_ids[0].cpu().numpy()
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)[8:-5].title()
    
    # Print out the image and the generated caption
    image = Image.open(image_path)
    return sentence, image

def PretrainedResNetBeamSearch(image_path, beam_width=5, **kwargs):
    """Wrapper that uses beam search instead of greedy"""
    # Load models and vocab (same as PretrainedResNet)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
    
    with open(kwargs.get('vocab_path', VOCAB_PATH), 'rb') as f:
        vocab = pickle.load(f)
    
    encoder = EncoderCNN(kwargs.get('embed_size', EMBED_SIZE)).eval()
    decoder = DecoderRNN(kwargs.get('embed_size', EMBED_SIZE),
                        kwargs.get('hidden_size', HIDDEN_SIZE),
                        len(vocab),
                        kwargs.get('num_layers', NUM_LAYERS))
    
    encoder.load_state_dict(torch.load(kwargs.get('encoder_path', ENCODER_PATH)))
    decoder.load_state_dict(torch.load(kwargs.get('decoder_path', DECODER_PATH)))
    
    # Generate caption with beam search
    image = load_image(image_path, transform)
    feature = encoder(image.to(device))
    sampled_ids = decoder.beam_search(feature, beam_width)
    
    # Convert to sentence
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
            
    sentence = ' '.join(sampled_caption)[8:-5].title()
    image = Image.open(image_path)
    return sentence, image

# Example usage
if __name__ == '__main__':
    image_path = 'path/to/your/image.jpg'
    caption, image = PretrainedResNet(image_path)
    print(caption)
    image.show()
