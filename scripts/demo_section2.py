#!/usr/bin/env python3
"""
Demonstration script for Section 2: Data Strategy & Student Simulation

This script shows how to:
1. Load and process mathematical datasets (GSM8K, OpenMathInstruct-2)
2. Create a structured problem bank with concept vectors and difficulty scores
3. Simulate diverse student populations and their learning behaviors
4. Generate RL training data from student-tutor interactions
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.dataset_loaders import load_combined_datasets, GSM8KLoader, OpenMathInstructLoader
from data_processing.problem_bank import ProblemBankCreator, ConceptVectorizer, DifficultyAnalyzer
from simulation.student_model import CognitiveStudentModel, StudentPopulation
from simulation.simulator import StudentSimulator, RLDataPreprocessor

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_dataset_loading():
    """Demonstrate loading and processing datasets"""
    print("=" * 60)
    print("1. DATASET LOADING DEMONSTRATION")
    print("=" * 60)
    
    # Load small sample for demonstration
    print("\n📚 Loading combined datasets (small sample)...")
    raw_data = load_combined_datasets(
        gsm8k_split="train", 
        openmath_max=50,  # Small sample for demo
        cache_dir="./demo_cache"
    )
    
    print(f"✅ Loaded {len(raw_data)} problems total")
    
    # Show sample problems
    print("\n📖 Sample problems:")
    for i, problem in enumerate(raw_data[:3]):
        print(f"\nProblem {i+1} ({problem['source_dataset']}):")
        print(f"Question: {problem['question'][:100]}...")
        print(f"Answer: {problem['answer']}")
        print(f"Step count: {problem.get('step_count', 'N/A')}")
    
    return raw_data


def demonstrate_problem_bank_creation(raw_data):
    """Demonstrate problem bank creation and concept analysis"""
    print("\n" + "=" * 60)
    print("2. PROBLEM BANK CREATION DEMONSTRATION")
    print("=" * 60)
    
    # Create concept vectorizer
    print("\n🧠 Initializing concept vectorizer...")
    vectorizer = ConceptVectorizer(vector_dim=50)
    
    # Demonstrate concept extraction
    sample_problem = raw_data[0]
    concepts = vectorizer.extract_concepts(
        sample_problem['question'], 
        sample_problem.get('solution', '')
    )
    print(f"📋 Sample problem concepts: {concepts}")
    
    # Create concept vector
    concept_vector = vectorizer.create_concept_vector(concepts)
    print(f"🔢 Concept vector shape: {concept_vector.shape}")
    print(f"📊 Non-zero dimensions: {np.sum(concept_vector > 0)}")
    
    # Create problem bank
    print("\n🏗️ Creating structured problem bank...")
    creator = ProblemBankCreator(vector_dim=50)
    processed_problems = creator.process_problems(raw_data)
    
    print(f"✅ Processed {len(processed_problems)} problems")
    
    # Analyze difficulty distribution
    difficulties = [p['difficulty_score'] for p in processed_problems]
    print(f"\n📈 Difficulty statistics:")
    print(f"  Mean: {np.mean(difficulties):.3f}")
    print(f"  Std: {np.std(difficulties):.3f}")
    print(f"  Range: {np.min(difficulties):.3f} - {np.max(difficulties):.3f}")
    
    # Show sample processed problem
    sample_processed = processed_problems[0]
    print(f"\n📚 Sample processed problem:")
    print(f"  ID: {sample_processed['id']}")
    print(f"  Concepts: {sample_processed['concepts']}")
    print(f"  Difficulty: {sample_processed['difficulty_score']:.3f}")
    print(f"  Estimated time: {sample_processed['estimated_time_minutes']} min")
    print(f"  Prerequisites: {sample_processed['prerequisite_concepts']}")
    
    return processed_problems


def demonstrate_student_simulation(processed_problems):
    """Demonstrate student model creation and simulation"""
    print("\n" + "=" * 60)
    print("3. STUDENT SIMULATION DEMONSTRATION")
    print("=" * 60)
    
    # Create individual student models
    print("\n👥 Creating diverse student models...")
    
    students = {
        'fast_learner': CognitiveStudentModel(
            student_id="fast_001",
            student_type="fast_learner",
            initial_knowledge_level=0.6,
            learning_rate=0.15,
            motivation_level=0.8
        ),
        'struggling_learner': CognitiveStudentModel(
            student_id="struggling_001", 
            student_type="struggling_learner",
            initial_knowledge_level=0.2,
            learning_rate=0.06,
            motivation_level=0.4
        ),
        'average': CognitiveStudentModel(
            student_id="average_001",
            student_type="average",
            initial_knowledge_level=0.35,
            learning_rate=0.1,
            motivation_level=0.6
        )
    }
    
    # Demonstrate problem attempts
    test_problem = processed_problems[0]
    print(f"\n🧪 Testing problem attempt with different students:")
    print(f"Problem difficulty: {test_problem['difficulty_score']:.3f}")
    
    for student_type, student in students.items():
        result = student.attempt_problem(test_problem)
        print(f"\n  {student_type.title()}:")
        print(f"    Success: {result['success']}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Time spent: {result['time_spent']}s")
        print(f"    Gave up: {result['gave_up']}")
        print(f"    Help requested: {result['help_requested']}")
    
    # Create population
    print(f"\n🌍 Creating student population...")
    population = StudentPopulation(num_students=100)  # Small for demo
    stats = population.get_population_stats()
    
    print(f"📊 Population statistics:")
    print(f"  Total students: {stats['total_students']}")
    print(f"  Type distribution: {stats['type_distribution']}")
    print(f"  Knowledge mean: {stats['knowledge_stats']['mean']:.3f}")
    print(f"  Motivation mean: {stats['motivation_stats']['mean']:.3f}")
    
    return population


def demonstrate_interaction_simulation(processed_problems, population):
    """Demonstrate full interaction simulation and RL data generation"""
    print("\n" + "=" * 60)
    print("4. INTERACTION SIMULATION DEMONSTRATION")
    print("=" * 60)
    
    # Create simulator
    print("\n🎯 Initializing student-tutor simulator...")
    simulator = StudentSimulator(processed_problems, population)
    
    # Generate sample interactions
    print(f"\n🎮 Generating interaction data (small sample)...")
    interaction_data = simulator.generate_interaction_data(num_sessions=20)  # Small for demo
    
    print(f"✅ Generated {len(interaction_data)} interactions")
    
    # Analyze interaction data
    successful_interactions = [i for i in interaction_data if i['result']['success']]
    gave_up_interactions = [i for i in interaction_data if i['result']['gave_up']]
    help_requested = [i for i in interaction_data if i['result']['help_requested']]
    
    print(f"\n📈 Interaction statistics:")
    print(f"  Success rate: {len(successful_interactions)/len(interaction_data)*100:.1f}%")
    print(f"  Give up rate: {len(gave_up_interactions)/len(interaction_data)*100:.1f}%")
    print(f"  Help request rate: {len(help_requested)/len(interaction_data)*100:.1f}%")
    
    # Show sample interaction
    sample_interaction = interaction_data[0]
    print(f"\n📝 Sample interaction:")
    print(f"  Student ID: {sample_interaction['student_id']}")
    print(f"  Student type: {sample_interaction['student_type']}")
    print(f"  Problem ID: {sample_interaction['problem_id']}")
    print(f"  Result: {sample_interaction['result']['success']}")
    print(f"  Reward: {sample_interaction['reward']:.2f}")
    print(f"  State shape: {sample_interaction['state'].shape}")
    print(f"  Action shape: {sample_interaction['action'].shape}")
    
    return interaction_data


def demonstrate_rl_preprocessing(interaction_data):
    """Demonstrate RL data preprocessing"""
    print("\n" + "=" * 60)
    print("5. RL DATA PREPROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Preprocess data
    print("\n⚙️ Preprocessing interaction data for RL training...")
    preprocessor = RLDataPreprocessor()
    training_data = preprocessor.prepare_training_data(interaction_data)
    
    print(f"✅ Preprocessed data shapes:")
    for key, array in training_data.items():
        print(f"  {key}: {array.shape}")
    
    # Get statistics
    stats = preprocessor.get_data_statistics(training_data)
    print(f"\n📊 Training data statistics:")
    print(f"  Number of samples: {stats['num_samples']}")
    print(f"  Success rate: {stats['success_rate']:.3f}")
    print(f"  Done rate: {stats['done_rate']:.3f}")
    print(f"  Reward statistics:")
    print(f"    Mean: {stats['reward_stats']['mean']:.2f}")
    print(f"    Std: {stats['reward_stats']['std']:.2f}")
    print(f"    Range: {stats['reward_stats']['min']:.2f} to {stats['reward_stats']['max']:.2f}")
    
    return training_data


def create_visualizations(processed_problems, interaction_data, training_data):
    """Create visualizations of the generated data"""
    print("\n" + "=" * 60)
    print("6. DATA VISUALIZATION")
    print("=" * 60)
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Section 2: Data Strategy & Student Simulation Results', fontsize=16)
    
    # 1. Problem difficulty distribution
    difficulties = [p['difficulty_score'] for p in processed_problems]
    axes[0, 0].hist(difficulties, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Problem Difficulty Distribution')
    axes[0, 0].set_xlabel('Difficulty Score')
    axes[0, 0].set_ylabel('Number of Problems')
    
    # 2. Concept distribution
    all_concepts = []
    for p in processed_problems:
        all_concepts.extend(p['concepts'])
    
    concept_counts = {}
    for concept in all_concepts:
        concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    concepts, counts = zip(*top_concepts)
    
    axes[0, 1].bar(range(len(concepts)), counts, color='lightcoral')
    axes[0, 1].set_title('Top Mathematical Concepts')
    axes[0, 1].set_xlabel('Concept')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xticks(range(len(concepts)))
    axes[0, 1].set_xticklabels(concepts, rotation=45, ha='right')
    
    # 3. Reward distribution
    rewards = training_data['rewards']
    axes[1, 0].hist(rewards, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('RL Reward Distribution')
    axes[1, 0].set_xlabel('Reward Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Student type performance
    student_performance = {}
    for interaction in interaction_data:
        student_type = interaction['student_type']
        success = interaction['result']['success']
        
        if student_type not in student_performance:
            student_performance[student_type] = []
        student_performance[student_type].append(success)
    
    types = []
    success_rates = []
    for student_type, successes in student_performance.items():
        types.append(student_type)
        success_rates.append(np.mean(successes))
    
    axes[1, 1].bar(types, success_rates, color='gold')
    axes[1, 1].set_title('Success Rate by Student Type')
    axes[1, 1].set_xlabel('Student Type')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_xticklabels(types, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'section2_demonstration.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualizations saved to: {output_dir / 'section2_demonstration.png'}")
    
    # Show plot
    plt.show()


def main():
    """Main demonstration function"""
    print("🚀 RL Math Tutor - Section 2 Demonstration")
    print("Data Strategy & Student Simulation Implementation")
    print("=" * 60)
    
    try:
        # Step 1: Load datasets
        raw_data = demonstrate_dataset_loading()
        
        # Step 2: Create problem bank
        processed_problems = demonstrate_problem_bank_creation(raw_data)
        
        # Step 3: Simulate students
        population = demonstrate_student_simulation(processed_problems)
        
        # Step 4: Generate interactions
        interaction_data = demonstrate_interaction_simulation(processed_problems, population)
        
        # Step 5: Preprocess for RL
        training_data = demonstrate_rl_preprocessing(interaction_data)
        
        # Step 6: Visualize results
        create_visualizations(processed_problems, interaction_data, training_data)
        
        print("\n" + "=" * 60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ Processed {len(processed_problems)} problems")
        print(f"✅ Created {len(population.students)} student models")
        print(f"✅ Generated {len(interaction_data)} student-tutor interactions")
        print(f"✅ Prepared {training_data['states'].shape[0]} RL training samples")
        print(f"📁 Results saved in: ./outputs/")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main() 