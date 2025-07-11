"""
Student-tutor interaction simulator for generating RL training data.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import logging
from .student_model import CognitiveStudentModel, StudentPopulation

logger = logging.getLogger(__name__)


class StudentSimulator:
    """Simulates student-tutor interactions to generate training data"""
    
    def __init__(self, problem_bank: List[Dict], student_population: StudentPopulation):
        self.problem_bank = problem_bank
        self.student_population = student_population
        
        # Group problems by difficulty for easier selection
        self.problems_by_difficulty = self._group_problems_by_difficulty()
        
    def _group_problems_by_difficulty(self) -> Dict[str, List[Dict]]:
        """Group problems into difficulty categories"""
        groups = {
            'easy': [],      # 0.0 - 0.3
            'medium': [],    # 0.3 - 0.6
            'hard': [],      # 0.6 - 1.0
        }
        
        for problem in self.problem_bank:
            difficulty = problem['difficulty_score']
            if difficulty <= 0.3:
                groups['easy'].append(problem)
            elif difficulty <= 0.6:
                groups['medium'].append(problem)
            else:
                groups['hard'].append(problem)
                
        logger.info(f"Problem distribution: Easy={len(groups['easy'])}, "
                   f"Medium={len(groups['medium'])}, Hard={len(groups['hard'])}")
        
        return groups
    
    def generate_interaction_data(self, num_sessions: int = 5000) -> List[Dict]:
        """Generate synthetic student-tutor interaction data"""
        logger.info(f"Generating {num_sessions} interaction sessions...")
        
        interaction_data = []
        
        for session_id in range(num_sessions):
            student = self.student_population.get_random_student()
            session_data = self._simulate_session(student, session_id)
            interaction_data.extend(session_data)
            
            if (session_id + 1) % 500 == 0:
                logger.info(f"Completed {session_id + 1}/{num_sessions} sessions")
                
        logger.info(f"Generated {len(interaction_data)} total interactions")
        return interaction_data
    
    def _simulate_session(self, student: CognitiveStudentModel, session_id: int) -> List[Dict]:
        """Simulate a complete tutoring session"""
        session_data = []
        problems_attempted = 0
        max_problems = np.random.randint(5, 20)  # Variable session length
        
        # Reset student session state
        student.current_session_problems = 0
        student.fatigue_level = 0.0
        
        while problems_attempted < max_problems and student.motivation_level > 0.2:
            # Get current state for RL training
            state = self._get_student_state(student)
            
            # Select problem using heuristic (will be replaced by RL agent)
            problem = self._select_problem_heuristic(student)
            
            if not problem:
                break
                
            # Student attempts problem
            result = student.attempt_problem(problem)
            
            # Get next state
            next_state = self._get_student_state(student)
            
            # Calculate reward
            reward = self._calculate_reward(result, student, problem)
            
            # Record interaction for RL training
            interaction = {
                'session_id': session_id,
                'student_id': student.student_id,
                'student_type': student.student_type,
                'state': state,
                'action': self._encode_action(problem),
                'problem_id': problem['id'],
                'result': result,
                'next_state': next_state,
                'reward': reward,
                'done': result['gave_up'] or student.motivation_level <= 0.2
            }
            
            session_data.append(interaction)
            problems_attempted += 1
            
            # Break if student gives up or motivation too low
            if result['gave_up'] or student.motivation_level <= 0.2:
                break
                
        return session_data
    
    def _select_problem_heuristic(self, student: CognitiveStudentModel) -> Optional[Dict]:
        """Simple heuristic for problem selection (to be replaced by RL)"""
        # Calculate student's average knowledge level
        avg_knowledge = np.mean(student.knowledge_state)
        
        # Select difficulty based on knowledge and recent performance
        recent_success_rate = (sum(student.recent_performance) / len(student.recent_performance) 
                              if student.recent_performance else 0.5)
        
        # Adaptive difficulty selection
        if recent_success_rate > 0.8:
            # Student doing well, increase difficulty
            target_difficulty = min(avg_knowledge + 0.2, 0.9)
        elif recent_success_rate < 0.3:
            # Student struggling, decrease difficulty
            target_difficulty = max(avg_knowledge - 0.2, 0.1)
        else:
            # Appropriate challenge level
            target_difficulty = avg_knowledge + np.random.uniform(-0.1, 0.1)
            
        # Find problems in target difficulty range
        target_range = 0.2
        suitable_problems = [
            p for p in self.problem_bank 
            if abs(p['difficulty_score'] - target_difficulty) <= target_range
        ]
        
        if not suitable_problems:
            # Fallback to any problem
            suitable_problems = self.problem_bank
            
        # Filter by concepts student is learning
        concept_aligned_problems = []
        for problem in suitable_problems:
            problem_vector = np.array(problem['concept_vector'])
            alignment = np.dot(student.knowledge_state, problem_vector)
            if alignment > 0.1:  # Some concept overlap
                concept_aligned_problems.append(problem)
                
        if concept_aligned_problems:
            return np.random.choice(concept_aligned_problems)
        else:
            return np.random.choice(suitable_problems) if suitable_problems else None
    
    def _get_student_state(self, student: CognitiveStudentModel) -> np.ndarray:
        """Get student state as numpy array for RL"""
        state_dict = student.get_current_state()
        
        # Combine knowledge state with other features
        knowledge_vector = state_dict['knowledge_state']  # 50 dims
        context_features = [
            state_dict['motivation_level'],
            state_dict['fatigue_level'], 
            state_dict['consecutive_failures'] / 10.0,  # normalize
            state_dict['recent_success_rate'],
            state_dict['current_session_problems'] / 20.0,  # normalize
            # Add more context features as needed
        ]
        
        # Pad context to fixed size (25 dims total)
        while len(context_features) < 25:
            context_features.append(0.0)
            
        return np.concatenate([knowledge_vector, context_features[:25]])
    
    def _encode_action(self, problem: Dict) -> np.ndarray:
        """Encode problem selection as action vector"""
        # For now, use problem's concept vector and difficulty as action encoding
        concept_vector = np.array(problem['concept_vector'])
        
        # Add difficulty and other problem features
        action_features = [
            problem['difficulty_score'],
            problem.get('estimated_time_minutes', 5) / 30.0,  # normalize
            len(problem.get('concepts', [])) / 10.0,  # normalize
        ]
        
        # Pad to make fixed size action vector
        while len(action_features) < 50:
            action_features.append(0.0)
            
        return np.concatenate([concept_vector, action_features[:47]])  # Total 97 dims
    
    def _calculate_reward(self, result: Dict, student: CognitiveStudentModel, 
                         problem: Dict) -> float:
        """Calculate reward for RL training"""
        reward = 0.0
        
        # Base reward for success/failure
        if result['success']:
            reward += 10.0
        else:
            reward -= 2.0
            
        # Bonus for maintaining high engagement
        if student.motivation_level > 0.7:
            reward += 3.0
        elif student.motivation_level > 0.5:
            reward += 1.0
        elif student.motivation_level < 0.3:
            reward -= 5.0
            
        # Penalty for giving up
        if result['gave_up']:
            reward -= 15.0
            
        # Bonus for appropriate challenge level (success probability in good range)
        confidence = result['confidence']
        if 0.4 <= confidence <= 0.7:
            reward += 5.0  # Optimal challenge zone
        elif confidence > 0.9:
            reward -= 1.0  # Too easy
        elif confidence < 0.2:
            reward -= 3.0  # Too hard
            
        # Bonus for help-seeking when appropriate
        if result['help_requested'] and confidence < 0.4:
            reward += 2.0  # Good to ask for help when struggling
            
        # Time efficiency bonus
        expected_time = problem.get('estimated_time_minutes', 5) * 60
        actual_time = result['time_spent']
        if actual_time < expected_time * 1.2 and result['success']:
            reward += 1.0  # Efficient solving
            
        # Learning progress bonus
        if student.consecutive_successes >= 2:
            reward += 2.0  # Building confidence
        elif student.consecutive_failures >= 3:
            reward -= 3.0  # Prevent frustration
            
        return reward


class RLDataPreprocessor:
    """Preprocesses simulation data for RL training"""
    
    def __init__(self):
        self.state_dim = 75  # Student knowledge (50) + context (25)
        self.action_dim = 97  # Problem encoding dimension
        
    def prepare_training_data(self, interaction_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Convert simulation data to RL training format"""
        logger.info(f"Preprocessing {len(interaction_data)} interactions for RL training...")
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for interaction in interaction_data:
            try:
                state = interaction['state']
                action = interaction['action']
                reward = interaction['reward']
                next_state = interaction['next_state']
                done = interaction['done']
                
                # Validate dimensions
                if len(state) == self.state_dim and len(action) == self.action_dim:
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)
                else:
                    logger.warning(f"Dimension mismatch: state={len(state)}, action={len(action)}")
                    
            except Exception as e:
                logger.warning(f"Error processing interaction: {e}")
                continue
                
        logger.info(f"Successfully preprocessed {len(states)} interactions")
        
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'dones': np.array(dones, dtype=bool)
        }
    
    def create_experience_buffer(self, training_data: Dict[str, np.ndarray], 
                                buffer_size: int = 100000) -> Dict[str, np.ndarray]:
        """Create experience replay buffer for RL training"""
        
        total_samples = len(training_data['states'])
        if total_samples <= buffer_size:
            return training_data
            
        # Randomly sample from the data
        indices = np.random.choice(total_samples, buffer_size, replace=False)
        
        return {
            'states': training_data['states'][indices],
            'actions': training_data['actions'][indices], 
            'rewards': training_data['rewards'][indices],
            'next_states': training_data['next_states'][indices],
            'dones': training_data['dones'][indices]
        }
    
    def get_data_statistics(self, training_data: Dict[str, np.ndarray]) -> Dict:
        """Get statistics about the training data"""
        return {
            'num_samples': len(training_data['states']),
            'state_stats': {
                'mean': np.mean(training_data['states'], axis=0),
                'std': np.std(training_data['states'], axis=0),
                'min': np.min(training_data['states'], axis=0),
                'max': np.max(training_data['states'], axis=0)
            },
            'reward_stats': {
                'mean': np.mean(training_data['rewards']),
                'std': np.std(training_data['rewards']),
                'min': np.min(training_data['rewards']),
                'max': np.max(training_data['rewards'])
            },
            'success_rate': np.mean(training_data['rewards'] > 0),
            'done_rate': np.mean(training_data['dones'])
        }


if __name__ == "__main__":
    # Test the simulator
    from ..data_processing.dataset_loaders import load_combined_datasets
    from ..data_processing.problem_bank import ProblemBankCreator
    
    # Load and process data
    raw_data = load_combined_datasets(openmath_max=100)
    creator = ProblemBankCreator()
    problem_bank = creator.process_problems(raw_data)
    
    # Create student population
    population = StudentPopulation(num_students=50)
    
    # Run simulation
    simulator = StudentSimulator(problem_bank, population)
    interaction_data = simulator.generate_interaction_data(num_sessions=100)
    
    # Preprocess for RL
    preprocessor = RLDataPreprocessor()
    training_data = preprocessor.prepare_training_data(interaction_data)
    stats = preprocessor.get_data_statistics(training_data)
    
    print(f"Generated training data with {stats['num_samples']} samples")
    print(f"Success rate: {stats['success_rate']:.2f}")
    print(f"Average reward: {stats['reward_stats']['mean']:.2f}") 