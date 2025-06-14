import torch
import json
import numpy as np
from typing import Dict, List, Tuple
from si_core.model.core import SI_Core
from si_core.utils.dataset import SI_Dataset
from si_core.utils.text_encoder import TextEncoder

class Tester:
    """Tester for SI-Core model."""
    def __init__(self,
                 model_path: str,
                 test_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize model
        self.model = SI_Core(state_dim=1024, embedding_dim=512)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        
        # Initialize dataset
        self.test_dataset = SI_Dataset(test_path)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=SI_Dataset.collate_fn
        )
        
        # Text encoder
        self.text_encoder = TextEncoder()
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance."""
        results = {
            'cosine_similarity': [],
            'response_length': [],
            'accuracy': 0,
            'total': 0
        }
        
        with torch.no_grad():
            for input_text, target_text in self.test_loader:
                # Get model response
                response = self.model(input_text[0])
                
                # Calculate cosine similarity
                target_emb = self.text_encoder.encode(target_text[0])
                response_emb = self.text_encoder.encode(response)
                sim = torch.nn.functional.cosine_similarity(
                    target_emb.unsqueeze(0),
                    response_emb.unsqueeze(0)
                ).item()
                results['cosine_similarity'].append(sim)
                
                # Track response length
                results['response_length'].append(len(response.split()))
                
                # Check if response is relevant
                if sim > 0.7:  # Threshold for relevance
                    results['accuracy'] += 1
                results['total'] += 1
        
        # Calculate metrics
        metrics = {
            'mean_cosine_similarity': np.mean(results['cosine_similarity']),
            'mean_response_length': np.mean(results['response_length']),
            'accuracy': results['accuracy'] / results['total']
        }
        
        return metrics
        
    def generate_responses(self, num_samples: int = 10) -> List[Dict[str, str]]:
        """Generate sample responses."""
        samples = []
        
        for i, (input_text, _) in enumerate(self.test_loader):
            if i >= num_samples:
                break
                
            # Get model response
            response = self.model(input_text[0])
            
            samples.append({
                'input': input_text[0],
                'response': response
            })
        
        return samples
        
    def test(self):
        """Run full test suite."""
        print("\nRunning tests...")
        
        # Evaluate metrics
        metrics = self.evaluate()
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Generate sample responses
        print("\nSample Responses:")
        samples = self.generate_responses()
        for sample in samples:
            print(f"\nInput: {sample['input']}")
            print(f"Response: {sample['response']}")

def main():
    # Initialize tester
    tester = Tester(
        model_path='checkpoints/si_core.pt',
        test_path='data/test.json'
    )
    
    # Run tests
    tester.test()

if __name__ == '__main__':
    main()
