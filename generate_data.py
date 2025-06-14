import json
import random
from typing import List, Dict

def generate_sample_data(num_samples: int = 100) -> List[Dict[str, str]]:
    """Generate sample training data."""
    data = []
    
    # Sample dialog patterns
    patterns = [
        ("What is the capital of {country}?", "The capital of {country} is {capital}"),
        ("Tell me about {animal}.", "{animal} is a {description} animal that {behavior}"),
        ("How does {technology} work?", "{technology} works by {explanation}"),
        ("What's your opinion on {topic}?", "I think {topic} is {opinion} because {reason}"),
    ]
    
    # Sample values
    countries = ["France", "Germany", "Italy", "Spain", "Japan"]
    capitals = ["Paris", "Berlin", "Rome", "Madrid", "Tokyo"]
    animals = ["dog", "cat", "elephant", "dolphin", "parrot"]
    technologies = ["AI", "blockchain", "quantum computing", "5G", "IoT"]
    topics = ["climate change", "space exploration", "education", "healthcare", "economy"]
    
    for _ in range(num_samples):
        pattern = random.choice(patterns)
        
        # Choose appropriate values based on pattern
        if "country" in pattern[0]:
            country = random.choice(countries)
            capital = capitals[countries.index(country)]
            input_text = pattern[0].format(country=country)
            target_text = pattern[1].format(country=country, capital=capital)
        elif "animal" in pattern[0]:
            animal = random.choice(animals)
            input_text = pattern[0].format(animal=animal)
            target_text = pattern[1].format(
                animal=animal,
                description=random.choice(["friendly", "intelligent", "social", "unique"]),
                behavior=random.choice(["lives in groups", "eats plants", "communicates with sounds", "is very active"])
            )
        elif "technology" in pattern[0]:
            technology = random.choice(technologies)
            input_text = pattern[0].format(technology=technology)
            target_text = pattern[1].format(
                technology=technology,
                explanation=random.choice([
                    "processing data using advanced algorithms",
                    "connecting devices in a network",
                    "solving complex problems",
                    "improving communication"
                ])
            )
        else:  # opinion pattern
            topic = random.choice(topics)
            input_text = pattern[0].format(topic=topic)
            target_text = pattern[1].format(
                topic=topic,
                opinion=random.choice(["important", "complex", "fascinating", "challenging"]),
                reason=random.choice([
                    "it affects many aspects of our lives",
                    "it requires careful consideration",
                    "it has significant implications",
                    "it's a topic worth exploring"
                ])
            )
        
        data.append({
            "input": input_text,
            "target": target_text
        })
    
    return data

import os

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate training data
    train_data = generate_sample_data(8000)
    with open(os.path.join('data', 'train.json'), 'w') as f:
        json.dump(train_data, f)
    
    # Generate validation data
    val_data = generate_sample_data(1000)
    with open(os.path.join('data', 'val.json'), 'w') as f:
        json.dump(val_data, f)
    
    # Generate test data
    test_data = generate_sample_data(1000)
    with open(os.path.join('data', 'test.json'), 'w') as f:
        json.dump(test_data, f)
    
    print("Data generation complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Print file locations
    print("\nData files created:")
    print(f"Training data: data/train.json")
    print(f"Validation data: data/val.json")
    print(f"Test data: data/test.json")

if __name__ == '__main__':
    main()
