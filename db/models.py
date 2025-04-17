import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class TestingJob(Base):
    __tablename__ = "testing_job"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    api_type = Column(String(50), nullable=False)
    model_id = Column(String(100), nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed, verifying, verified
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    current_tree_level = Column(Integer, default=0)  # Track which level of the tree we're processing
    
    # Relationships
    responses = relationship("ModelResponse", back_populates="job", cascade="all, delete-orphan")
    tree_paths = relationship("TreePathNode", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<TestingJob {self.model_name}>'


class ModelResponse(Base):
    __tablename__ = "model_response"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('testing_job.id', ondelete='CASCADE'), nullable=False)
    question_id = Column(String(20), nullable=False)  # e.g. "question_1"
    raw_response = Column(Text, nullable=False)  # The actual response text
    category = Column(String(100), nullable=True)  # Categorization result, nullable until processed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_flagged = Column(Boolean, default=False)  # Indicates if this response has been flagged for errors
    corrected_category = Column(String(100), nullable=True)  # The manually corrected category if flagged
    flagged_at = Column(DateTime, nullable=True)  # When the response was flagged
    
    # Tree-related fields
    tree_level = Column(Integer, default=0)  # Level in the decision tree (0=root question)
    parent_path = Column(String(255), nullable=True)  # Path from root to parent (null for root level)
    path_context = Column(String(255), nullable=True)  # Full path including this response
    
    # Option shuffling fields
    option_mapping = Column(Text, nullable=True)  # JSON mapping of original options to shuffled positions
    shuffled_options = Column(Text, nullable=True)  # The actual shuffled options presented
    
    # Relationships
    job = relationship("TestingJob", back_populates="responses")
    
    def __repr__(self):
        return f'<ModelResponse {self.job_id}:{self.question_id}>'


class CategoryCount(Base):
    __tablename__ = "category_count"
    
    id = Column(Integer, primary_key=True)
    question_id = Column(String(20), nullable=False)  # e.g. "question_1"
    category = Column(String(100), nullable=False)  # e.g. "refusal", "Blue", etc.
    model_name = Column(String(100), nullable=False)  # model name for easy lookups
    count = Column(Integer, default=0)  # number of times this category appears
    
    # Tree-related fields
    tree_level = Column(Integer, default=0)  # Level in the decision tree
    parent_path = Column(String(255), nullable=True)  # Path context (for level 1+)
    
    # Add unique constraints for tree-based data
    __table_args__ = (
        # This index includes parent_path to allow same categories in different tree paths
        UniqueConstraint('question_id', 'category', 'model_name', 'tree_level', 'parent_path', 
                        name='_question_category_model_path_uc'),
    )
    
    def __repr__(self):
        return f'<CategoryCount {self.model_name}:{self.question_id}:{self.category}:{self.parent_path}>'


class TreePathNode(Base):
    """Tracks active paths in the decision tree"""
    __tablename__ = "tree_path_node"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('testing_job.id', ondelete='CASCADE'), nullable=False)
    question_id = Column(String(20), nullable=False)  # e.g. "question_1" 
    tree_level = Column(Integer, nullable=False)  # Level in the tree (0=root)
    parent_path = Column(String(255), nullable=True)  # Path from root to parent (null for root)
    category = Column(String(100), nullable=False)  # Branch category
    is_active = Column(Boolean, default=True)  # Whether this path should be sampled
    sample_count = Column(Integer, default=0)  # Number of samples taken for this path
    
    # Relationships
    job = relationship("TestingJob", back_populates="tree_paths")
    
    # Unique constraint to prevent duplicate paths
    __table_args__ = (
        UniqueConstraint('job_id', 'question_id', 'tree_level', 'parent_path', 'category',
                        name='_job_path_category_uc'),
    )
    
    def __repr__(self):
        path_str = f"{self.parent_path}→{self.category}" if self.parent_path else self.category
        return f'<TreePathNode {self.job_id}:{self.question_id}:{path_str}>'


class TestStatus(Base):
    __tablename__ = "test_status"
    
    id = Column(Integer, primary_key=True)  # Will only ever be one row with id=1
    is_running = Column(Boolean, default=False)  # Whether a test is currently running
    current_model = Column(String(100), nullable=True)  # Current model being tested
    job_id = Column(Integer, nullable=True)  # Current job ID
    started_at = Column(DateTime, nullable=True)  # When the current test started
    
    def __repr__(self):
        return f'<TestStatus is_running={self.is_running} model={self.current_model}>'