"""
Cognitive student models for simulating realistic learning behaviors.
"""

import numpy as np
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CognitiveStudentModel:
    """Simulates a student's cognitive processes and learning behavior"""
    
    def __init__(self, 
                 student_id: str,
                 initial_knowledge_level: float = 0.3,
                 learning_rate: float = 0.1,
                 attention_span: int = 10,
                 motivation_level: float = 0.7,
                 student_type: str = "average"):
        
        self.student_id = student_id
        self.student_type = student_type
        
        # Core cognitive parameters
        self.knowledge_state = np.random.uniform(0.1, initial_knowledge_level, 50)
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.attention_span = attention_span
        self.base_motivation = motivation_level
        self.motivation_level = motivation_level
        
        # Dynamic state variables
        self.current_session_problems = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.recent_performance = deque(maxlen=5)
        self.fatigue_level = 0.0
        self.confidence_level = 0.5
        
        # Learning style parameters
        self.prefers_visual = np.random.uniform(0.3, 0.8)
        self.prefers_step_by_step = np.random.uniform(0.4, 0.9)
        self.challenge_tolerance = np.random.uniform(0.2, 0.8)
        self.help_seeking_tendency = np.random.uniform(0.1, 0.7)
        
        # Performance tracking
        self.total_problems_attempted = 0
        self.total_problems_solved = 0
        
        self._apply_student_type_modifiers()
        
    def _apply_student_type_modifiers(self):
        """Apply modifiers based on student type"""
        modifiers = {
            'fast_learner': {
                'learning_rate_multiplier': 1.5,
                'knowledge_boost': 0.2,
                'motivation_boost': 0.1,
                'attention_boost': 5,
                'challenge_tolerance_boost': 0.2
            },
            'struggling_learner': {
                'learning_rate_multiplier': 0.6,
                'knowledge_boost': -0.1,
                'motivation_boost': -0.2,
                'attention_boost': -3,
                'challenge_tolerance_boost': -0.3
            }
        }
        
        if self.student_type in modifiers:
            mods = modifiers[self.student_type]
            
            self.learning_rate *= mods.get('learning_rate_multiplier', 1.0)
            self.knowledge_state += mods.get('knowledge_boost', 0.0)
            self.knowledge_state = np.clip(self.knowledge_state, 0.0, 1.0)
            self.motivation_level += mods.get('motivation_boost', 0.0)
            self.motivation_level = max(0.1, min(1.0, self.motivation_level))
    
    def attempt_problem(self, problem: Dict) -> Dict:
        """Simulate student's attempt at solving a problem"""
        self.total_problems_attempted += 1
        
        # Calculate knowledge alignment with problem concepts
        problem_vector = np.array(problem['concept_vector'])
        knowledge_alignment = np.dot(self.knowledge_state, problem_vector)
        
        # Apply difficulty adjustment
        difficulty_factor = max(0.1, 1.2 - problem['difficulty_score'])
        
        # Apply current state effects
        fatigue_penalty = self.fatigue_level * 0.3
        motivation_bonus = (self.motivation_level - 0.5) * 0.2
        
        # Calculate success probability
        base_probability = knowledge_alignment * difficulty_factor
        adjusted_probability = base_probability + motivation_bonus - fatigue_penalty
        success_probability = max(0.05, min(0.95, adjusted_probability))
        
        # Determine success
        success = np.random.random() < success_probability
        
        # Calculate time spent
        time_spent = self._calculate_time_spent(problem, success, success_probability)
        
        # Check if student gives up
        gave_up = self._check_if_gave_up(problem, success_probability)
        
        # Check if student requests help
        help_requested = self._check_help_request(problem, success, success_probability)
        
        result = {
            'success': success and not gave_up,
            'time_spent': time_spent,
            'confidence': success_probability,
            'gave_up': gave_up,
            'help_requested': help_requested,
            'problem_id': problem['id']
        }
        
        self._update_state_after_attempt(problem, result)
        return result
    
    def _calculate_time_spent(self, problem: Dict, success: bool, confidence: float) -> int:
        """Calculate time spent on problem in seconds"""
        base_time = problem.get('estimated_time_minutes', 3) * 60
        
        if success:
            time_multiplier = 0.7 if confidence > 0.7 else 1.0
        else:
            time_multiplier = 1.5 if confidence > 0.5 else 1.2
            
        fatigue_multiplier = 1.0 + (self.fatigue_level * 0.3)
        total_time = base_time * time_multiplier * fatigue_multiplier
        total_time *= np.random.uniform(0.8, 1.2)
        
        return max(30, int(total_time))
    
    def _check_if_gave_up(self, problem: Dict, success_probability: float) -> bool:
        """Check if student gives up on the problem"""
        give_up_probability = 0.0
        
        if self.motivation_level < 0.3:
            give_up_probability += 0.3
        
        give_up_probability += self.fatigue_level * 0.2
        
        if success_probability < 0.2:
            give_up_probability += 0.4
            
        if self.consecutive_failures >= 3:
            give_up_probability += 0.3
            
        return np.random.random() < min(0.6, give_up_probability)
    
    def _check_help_request(self, problem: Dict, success: bool, confidence: float) -> bool:
        """Check if student requests help"""
        help_probability = self.help_seeking_tendency * 0.5
        
        if confidence < 0.3:
            help_probability += 0.3
            
        if len(self.recent_performance) >= 2:
            recent_success_rate = sum(self.recent_performance) / len(self.recent_performance)
            if recent_success_rate < 0.4:
                help_probability += 0.2
                
        return np.random.random() < min(0.8, help_probability)
    
    def _update_state_after_attempt(self, problem: Dict, result: Dict):
        """Update student's internal state after problem attempt"""
        success = result['success']
        gave_up = result['gave_up']
        
        problem_vector = np.array(problem['concept_vector'])
        
        if success and not gave_up:
            learning_amount = self.learning_rate * (1 - self.fatigue_level)
            knowledge_update = learning_amount * problem_vector * 0.1
            self.knowledge_state += knowledge_update
            self.knowledge_state = np.clip(self.knowledge_state, 0, 1)
            
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            self.motivation_level = min(1.0, self.motivation_level + 0.02)
            self.total_problems_solved += 1
            
        elif not gave_up:
            learning_amount = self.learning_rate * 0.3 * (1 - self.fatigue_level)
            knowledge_update = learning_amount * problem_vector * 0.02
            self.knowledge_state += knowledge_update
            
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            if self.consecutive_failures > 2:
                self.motivation_level = max(0.1, self.motivation_level - 0.05)
        else:
            self.consecutive_failures += 1
            self.motivation_level = max(0.1, self.motivation_level - 0.1)
        
        self.current_session_problems += 1
        if self.current_session_problems > self.attention_span:
            self.fatigue_level = min(1.0, self.fatigue_level + 0.1)
            
        self.recent_performance.append(1 if success and not gave_up else 0)
    
    def get_current_state(self) -> Dict:
        """Get current student state for RL agent"""
        recent_success_rate = (sum(self.recent_performance) / len(self.recent_performance) 
                              if self.recent_performance else 0.5)
        
        return {
            'knowledge_state': self.knowledge_state.copy(),
            'motivation_level': self.motivation_level,
            'fatigue_level': self.fatigue_level,
            'consecutive_failures': self.consecutive_failures,
            'recent_success_rate': recent_success_rate,
            'current_session_problems': self.current_session_problems,
            'student_type': self.student_type
        }


class StudentPopulation:
    """Manages a diverse population of student models"""
    
    def __init__(self, num_students: int = 1000):
        self.num_students = num_students
        self.students = []
        self._create_student_population()
        
    def _create_student_population(self):
        """Create diverse student population"""
        student_type_distribution = {
            'average': 0.40,
            'fast_learner': 0.20,
            'struggling_learner': 0.20,
            'anxious_learner': 0.10,
            'confident_learner': 0.10
        }
        
        for i in range(self.num_students):
            student_type = np.random.choice(
                list(student_type_distribution.keys()),
                p=list(student_type_distribution.values())
            )
            
            student = self._create_student_by_type(f"student_{i:04d}", student_type)
            self.students.append(student)
            
    def _create_student_by_type(self, student_id: str, student_type: str) -> CognitiveStudentModel:
        """Create student with specific type characteristics"""
        base_params = {
            'student_id': student_id,
            'student_type': student_type
        }
        
        if student_type == 'fast_learner':
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.4, 0.7),
                'learning_rate': np.random.uniform(0.12, 0.18),
                'attention_span': np.random.randint(12, 20),
                'motivation_level': np.random.uniform(0.7, 0.9)
            }
        elif student_type == 'struggling_learner':
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.1, 0.3),
                'learning_rate': np.random.uniform(0.04, 0.08),
                'attention_span': np.random.randint(4, 8),
                'motivation_level': np.random.uniform(0.3, 0.6)
            }
        else:  # average, anxious_learner, confident_learner
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.25, 0.45),
                'learning_rate': np.random.uniform(0.08, 0.12),
                'attention_span': np.random.randint(8, 14),
                'motivation_level': np.random.uniform(0.5, 0.8)
            }
            
        return CognitiveStudentModel(**params)
    
    def get_random_student(self) -> CognitiveStudentModel:
        """Get a random student from the population"""
        return np.random.choice(self.students) 