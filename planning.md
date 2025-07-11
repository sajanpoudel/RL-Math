# RL Math Tutor Project Plan

This document outlines the plan for developing an adaptive math tutoring system powered by Reinforcement Learning.

*Note: The initial implementation will focus on the development and training of the Reinforcement Learning model and the LLM Service (Sections 2, 3, 4). The frontend and backend components will be implemented in a later phase.*

## 1. System Architecture Overview

The system is designed as a modular, cloud-native application. The core idea is to separate the concerns of the user interface, the backend logic, the RL-based pedagogical decisions, and the LLM-powered content generation.

The architecture consists of five primary components:

*   **Frontend (React + Vite):** A responsive web interface where students interact with problems. It communicates with the backend via a REST API. It's responsible for rendering math equations (using KaTeX/MathJax), capturing user input, and displaying feedback.
*   **Backend (FastAPI):** The central nervous system of the application. This Python-based server exposes API endpoints to the frontend, manages user sessions, queries the database, and orchestrates calls to the RL Agent and the LLM Service.
*   **RL Pedagogical Agent (Stable Baselines3):** A standalone service that embodies the teaching strategy. When the backend needs a new problem for a student, it sends the student's current state to this agent. The agent's policy (a trained neural network) returns an action (e.g., "select a medium-difficulty problem on 'factoring quadratics'").
*   **LLM Content Service (Hugging Face TRL):** A service that hosts the fine-tuned LLM. It receives requests from the backend to generate personalized hints, step-by-step explanations, or even new, similar problems based on a student's context.
*   **Data & Logging Pipeline (PostgreSQL + Redis):**
    *   PostgreSQL: The primary database for persistent data, such as user accounts, the problem bank, and historical student performance logs.
    *   Redis: An in-memory data store used for caching, managing user sessions, and, crucially, as a real-time replay buffer for online reinforcement learning.

## 2. Data Strategy & Student Simulation

Since the provided datasets (GSM8K, OpenMathInstruct-2) are not sequential user logs, we must first create a high-quality problem bank and then simulate student interactions to generate the initial training data for the RL agent.

### Step 1: Problem Bank Creation

**2.1.1 Dataset Processing Pipeline**

**Primary Datasets:**
- **GSM8K Dataset:** 8.5K grade school math word problems with step-by-step solutions
  - Structure: Each entry contains `question` (string) and `answer` (string) 
  - Problems typically require 2-8 reasoning steps
  - Covers arithmetic, basic algebra, word problems, percentages, ratios
  
- **OpenMathInstruct-2 Dataset:** 14M mathematical instruction tuning pairs
  - Structure: Each entry contains `problem` (string), `generated_solution` (string), `expected_answer` (string), and `problem_source` (string)
  - Covers elementary to advanced mathematics including calculus, algebra, geometry
  - Solutions use code-interpreter format with Python code blocks

**Processing Steps:**

1. **Data Cleaning & Validation:**
   ```python
   def validate_problem_entry(entry):
       """Validate and clean individual problem entries"""
       # Check for complete question and answer
       # Validate mathematical notation consistency
       # Ensure solution steps are logical
       # Remove malformed or incomplete entries
       pass
   ```

2. **Concept Extraction & Tagging:**
   ```python
   def extract_mathematical_concepts(problem_text, solution_text):
       """Extract mathematical concepts using NLP and pattern matching"""
       concepts = []
       
       # Pattern-based concept detection
       concept_patterns = {
           'arithmetic': ['addition', 'subtraction', 'multiplication', 'division'],
           'fractions': ['fraction', 'numerator', 'denominator', '/', 'parts'],
           'percentages': ['percent', '%', 'percentage of'],
           'algebra': ['equation', 'variable', 'solve for', 'x ='],
           'geometry': ['area', 'perimeter', 'volume', 'triangle', 'circle'],
           'word_problems': ['if', 'how much', 'how many', 'total cost'],
           'ratios': ['ratio', 'proportion', 'per', 'rate'],
           'statistics': ['average', 'mean', 'median', 'probability']
       }
       
       # NLP-based concept extraction using spaCy
       # Mathematical expression parsing
       # Solution step analysis
       
       return concepts
   ```

3. **Difficulty Assessment:**
   ```python
   def calculate_difficulty_score(problem, solution):
       """Calculate difficulty score based on multiple factors"""
       factors = {
           'step_count': count_solution_steps(solution),
           'concept_complexity': assess_concept_difficulty(problem),
           'numerical_complexity': analyze_numbers(problem),
           'language_complexity': assess_reading_level(problem),
           'prerequisite_concepts': count_prerequisite_concepts(problem)
       }
       
       # Weighted combination of factors
       difficulty = (
           0.25 * normalize_step_count(factors['step_count']) +
           0.30 * factors['concept_complexity'] +
           0.20 * factors['numerical_complexity'] +
           0.15 * factors['language_complexity'] +
           0.10 * factors['prerequisite_concepts']
       )
       
       return min(max(difficulty, 0.1), 1.0)  # Normalize to [0.1, 1.0]
   ```

4. **Concept Vector Creation:**
   Based on research into mathematical concept modeling, we'll create dense vector representations:
   
   ```python
   class ConceptVectorizer:
       def __init__(self, vector_dim=50):
           self.vector_dim = vector_dim
           self.concept_embeddings = {}
           
       def create_concept_vector(self, concepts):
           """Create vector representation of mathematical concepts"""
           # Initialize with zero vector
           vector = np.zeros(self.vector_dim)
           
           # Map concepts to vector dimensions
           concept_map = {
               'arithmetic': [0, 4],      # Basic operations
               'fractions': [5, 9],       # Fraction operations  
               'percentages': [10, 12],   # Percentage calculations
               'algebra': [13, 20],       # Algebraic thinking
               'geometry': [21, 28],      # Spatial reasoning
               'word_problems': [29, 35], # Reading comprehension + math
               'ratios': [36, 40],        # Proportional reasoning
               'statistics': [41, 49]     # Data analysis
           }
           
           for concept in concepts:
               if concept in concept_map:
                   start, end = concept_map[concept]
                   vector[start:end+1] = 1.0
                   
           return vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
   ```

**Database Schema:**
```sql
CREATE TABLE problems (
    id SERIAL PRIMARY KEY,
    source_dataset VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    solution TEXT NOT NULL,
    answer VARCHAR(500) NOT NULL,
    concepts TEXT[] NOT NULL,
    concept_vector FLOAT[] NOT NULL,
    difficulty_score FLOAT NOT NULL CHECK (difficulty_score >= 0.1 AND difficulty_score <= 1.0),
    estimated_time_minutes INTEGER,
    prerequisite_concepts TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_problems_difficulty ON problems(difficulty_score);
CREATE INDEX idx_problems_concepts ON problems USING GIN(concepts);
CREATE INDEX idx_problems_source ON problems(source_dataset);
```

### Step 2: Student Simulation Framework

**2.2.1 Cognitive Student Model**

Based on educational psychology research, we'll model students with the following internal state:

```python
class CognitiveStudentModel:
    def __init__(self, initial_knowledge_level=0.3, learning_rate=0.1, 
                 attention_span=10, motivation_level=0.7):
        # Core cognitive parameters
        self.knowledge_state = np.random.uniform(0.1, initial_knowledge_level, 50)  # Concept mastery
        self.learning_rate = learning_rate
        self.attention_span = attention_span  # Problems before fatigue
        self.motivation_level = motivation_level
        
        # Dynamic state variables
        self.current_session_problems = 0
        self.consecutive_failures = 0
        self.recent_performance = deque(maxlen=5)
        self.fatigue_level = 0.0
        
        # Learning style parameters
        self.prefers_visual = np.random.uniform(0.3, 0.8)
        self.prefers_step_by_step = np.random.uniform(0.4, 0.9)
        self.challenge_tolerance = np.random.uniform(0.2, 0.8)
        
    def attempt_problem(self, problem):
        """Simulate student's attempt at solving a problem"""
        # Calculate probability of success based on knowledge alignment
        knowledge_alignment = np.dot(self.knowledge_state, problem.concept_vector)
        
        # Apply difficulty adjustment
        difficulty_factor = max(0.1, 1.0 - problem.difficulty_score)
        
        # Apply fatigue and motivation effects
        fatigue_penalty = self.fatigue_level * 0.3
        motivation_bonus = self.motivation_level * 0.2
        
        # Calculate success probability
        base_probability = knowledge_alignment * difficulty_factor
        adjusted_probability = base_probability + motivation_bonus - fatigue_penalty
        
        success = np.random.random() < max(0.05, min(0.95, adjusted_probability))
        
        # Update internal state based on attempt
        self._update_state_after_attempt(problem, success)
        
        return {
            'success': success,
            'time_spent': self._calculate_time_spent(problem, success),
            'confidence': adjusted_probability,
            'gave_up': self._check_if_gave_up(problem),
            'help_requested': self._check_help_request(problem, success)
        }
        
    def _update_state_after_attempt(self, problem, success):
        """Update student's internal state after problem attempt"""
        # Update knowledge state
        if success:
            # Strengthen knowledge in relevant concepts
            learning_amount = self.learning_rate * (1 - self.fatigue_level)
            self.knowledge_state += learning_amount * problem.concept_vector * 0.1
            self.knowledge_state = np.clip(self.knowledge_state, 0, 1)
            
            # Reset consecutive failures
            self.consecutive_failures = 0
            
            # Slight motivation boost
            self.motivation_level = min(1.0, self.motivation_level + 0.02)
        else:
            # Minimal learning from failure (but some occurs)
            learning_amount = self.learning_rate * 0.3 * (1 - self.fatigue_level)
            self.knowledge_state += learning_amount * problem.concept_vector * 0.02
            
            # Track consecutive failures
            self.consecutive_failures += 1
            
            # Motivation impact depends on challenge tolerance
            if self.consecutive_failures > self.challenge_tolerance * 5:
                self.motivation_level = max(0.1, self.motivation_level - 0.05)
        
        # Update fatigue and session tracking
        self.current_session_problems += 1
        if self.current_session_problems > self.attention_span:
            self.fatigue_level = min(1.0, self.fatigue_level + 0.1)
            
        # Track recent performance
        self.recent_performance.append(1 if success else 0)
```

**2.2.2 Student Diversity Modeling**

To create realistic training data, we'll simulate diverse student populations:

```python
class StudentPopulation:
    def __init__(self, num_students=1000):
        self.students = []
        
        # Create diverse student profiles
        for i in range(num_students):
            student_type = np.random.choice([
                'fast_learner', 'steady_learner', 'struggling_learner',
                'high_achiever', 'anxious_learner', 'confident_learner'
            ])
            
            student = self._create_student_by_type(student_type)
            self.students.append(student)
    
    def _create_student_by_type(self, student_type):
        """Create student with specific learning characteristics"""
        if student_type == 'fast_learner':
            return CognitiveStudentModel(
                initial_knowledge_level=0.5,
                learning_rate=0.15,
                attention_span=15,
                motivation_level=0.8
            )
        elif student_type == 'struggling_learner':
            return CognitiveStudentModel(
                initial_knowledge_level=0.2,
                learning_rate=0.05,
                attention_span=6,
                motivation_level=0.4
            )
        # ... define other student types
```

**2.2.3 Simulation Data Generation**

```python
class StudentSimulator:
    def __init__(self, problem_bank, student_population):
        self.problem_bank = problem_bank
        self.student_population = student_population
        
    def generate_interaction_data(self, num_sessions=5000):
        """Generate synthetic student-tutor interaction data"""
        interaction_data = []
        
        for session_id in range(num_sessions):
            student = np.random.choice(self.student_population.students)
            session_data = self._simulate_session(student, session_id)
            interaction_data.extend(session_data)
            
        return interaction_data
    
    def _simulate_session(self, student, session_id):
        """Simulate a complete tutoring session"""
        session_data = []
        problems_attempted = 0
        max_problems = np.random.randint(5, 20)  # Variable session length
        
        while problems_attempted < max_problems and student.motivation_level > 0.2:
            # Current state for RL training
            state = self._get_student_state(student)
            
            # Simple problem selection (will be replaced by RL agent)
            problem = self._select_problem_heuristic(student)
            
            # Student attempts problem
            result = student.attempt_problem(problem)
            
            # Record interaction for RL training
            interaction = {
                'session_id': session_id,
                'student_state': state,
                'problem_id': problem.id,
                'action': self._encode_action(problem),
                'result': result,
                'next_state': self._get_student_state(student),
                'reward': self._calculate_reward(result, student)
            }
            
            session_data.append(interaction)
            problems_attempted += 1
            
            # Break if student gives up
            if result['gave_up']:
                break
                
        return session_data
    
    def _calculate_reward(self, result, student):
        """Calculate reward for RL training"""
        reward = 0
        
        # Base reward for success/failure
        if result['success']:
            reward += 10
        else:
            reward -= 2
            
        # Bonus for maintaining engagement
        if student.motivation_level > 0.6:
            reward += 2
            
        # Penalty for excessive difficulty (leading to giving up)
        if result['gave_up']:
            reward -= 15
            
        # Bonus for appropriate challenge level
        if 0.4 <= result['confidence'] <= 0.7:
            reward += 3
            
        return reward
```

**Database Schema for Simulation Data:**
```sql
CREATE TABLE student_interactions (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    problem_id INTEGER NOT NULL REFERENCES problems(id),
    student_state JSONB NOT NULL,
    action_taken JSONB NOT NULL,
    result JSONB NOT NULL,
    reward FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_interactions_session ON student_interactions(session_id);
CREATE INDEX idx_interactions_student ON student_interactions(student_id);
```

### Step 3: Initial RL Training Data Preparation

**2.3.1 State-Action-Reward Tuple Generation**

```python
class RLDataPreprocessor:
    def __init__(self):
        self.state_dim = 75  # Student knowledge (50) + session context (25)
        self.action_dim = 100  # Problem selection space
        
    def prepare_training_data(self, interaction_data):
        """Convert simulation data to RL training format"""
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for interaction in interaction_data:
            state = self._encode_state(interaction['student_state'])
            action = self._encode_action(interaction['action_taken'])
            reward = interaction['reward']
            next_state = self._encode_state(interaction.get('next_state', interaction['student_state']))
            done = interaction['result'].get('gave_up', False)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones)
        }
        
    def _encode_state(self, student_state):
        """Encode student state for RL agent"""
        # Combine knowledge state with session context
        knowledge_vector = student_state['knowledge_state']  # 50 dims
        context_vector = [
            student_state.get('motivation_level', 0.5),
            student_state.get('fatigue_level', 0.0),
            student_state.get('consecutive_failures', 0) / 10.0,  # normalize
            len(student_state.get('recent_performance', [])) / 5.0,  # normalize
            sum(student_state.get('recent_performance', [])) / max(1, len(student_state.get('recent_performance', [1]))),  # success rate
            student_state.get('current_session_problems', 0) / 20.0,  # normalize
            # ... additional context features (19 more to reach 25)
        ]
        
        return np.concatenate([knowledge_vector, context_vector])
```

**Expected Outcomes:**
- Problem bank with ~20K high-quality problems from GSM8K and OpenMathInstruct-2
- Rich concept vectorization covering 8 major mathematical domains
- Simulation data representing ~100K student-problem interactions
- Diverse student models representing various learning patterns
- Clean state-action-reward tuples ready for RL training

This comprehensive approach ensures our RL agent will have rich, realistic training data that captures the complexity of human learning while providing a solid foundation for the pedagogical decision-making system. 