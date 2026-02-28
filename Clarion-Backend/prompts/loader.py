"""
Prompt Loader - Load and manage versioned prompt templates.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from jinja2 import Template, Environment, BaseLoader

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Load and manage versioned prompt templates.
    
    Supports:
    - Versioned prompt storage
    - Jinja2 template rendering
    - Hot-reload capability (for development)
    - Template caching
    - Metadata extraction
    
    Example:
        loader = PromptLoader(version="v1", prompts_dir="./prompts")
        
        # Load and render a prompt
        prompt = loader.load(
            "concept_extraction",
            context=chunk_text,
            max_concepts=10
        )
    """
    
    def __init__(
        self,
        prompts_dir: str = "./prompts",
        version: str = "v1",
        auto_reload: bool = False,
        cache_templates: bool = True
    ):
        """
        Initialize prompt loader.
        
        Args:
            prompts_dir: Base directory for prompts
            version: Prompt version to use (subdirectory name)
            auto_reload: Reload templates on each load (dev mode)
            cache_templates: Cache compiled templates
        """
        self.prompts_dir = Path(prompts_dir)
        self.version = version
        self.version_dir = self.prompts_dir / version
        self.auto_reload = auto_reload
        self.cache_templates = cache_templates
        
        self._template_cache: Dict[str, Template] = {}
        self._file_mtimes: Dict[str, float] = {}
        
        # Ensure version directory exists
        if not self.version_dir.exists():
            logger.warning(f"Prompt version directory not found: {self.version_dir}")
    
    def load(
        self,
        prompt_name: str,
        **variables
    ) -> str:
        """
        Load and render a prompt template.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            **variables: Template variables for rendering
        
        Returns:
            Rendered prompt string
        
        Raises:
            FileNotFoundError: If prompt file not found
            ValueError: If template rendering fails
        """
        template = self._get_template(prompt_name)
        
        try:
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Failed to render prompt '{prompt_name}': {e}")
            raise ValueError(f"Template rendering failed: {e}") from e
    
    def _get_template(self, prompt_name: str) -> Template:
        """
        Get compiled template, using cache if enabled.
        
        Args:
            prompt_name: Name of the prompt
        
        Returns:
            Compiled Jinja2 template
        """
        file_path = self._get_file_path(prompt_name)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt not found: {file_path}")
        
        cache_key = str(file_path)
        
        # Check if we need to reload
        should_reload = self.auto_reload
        
        if not should_reload and self.cache_templates and cache_key in self._template_cache:
            # Check if file modified
            current_mtime = file_path.stat().st_mtime
            if self._file_mtimes.get(cache_key) != current_mtime:
                should_reload = True
        
        if should_reload or cache_key not in self._template_cache:
            # Load and compile template
            template_content = file_path.read_text(encoding="utf-8")
            template = Template(template_content)
            
            if self.cache_templates:
                self._template_cache[cache_key] = template
                self._file_mtimes[cache_key] = file_path.stat().st_mtime
            
            return template
        
        return self._template_cache[cache_key]
    
    def _get_file_path(self, prompt_name: str) -> Path:
        """Get full file path for prompt."""
        # Add .txt extension if not present
        if not prompt_name.endswith(".txt"):
            prompt_name += ".txt"
        
        return self.version_dir / prompt_name
    
    def list_available(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available prompts for current version.
        
        Returns:
            Dictionary mapping prompt names to metadata
        """
        if not self.version_dir.exists():
            return {}
        
        prompts = {}
        
        for file_path in self.version_dir.glob("*.txt"):
            prompt_name = file_path.stem
            
            # Try to extract metadata from first comment block
            try:
                content = file_path.read_text(encoding="utf-8")
                metadata = self._extract_metadata(content)
                
                prompts[prompt_name] = {
                    "name": prompt_name,
                    "version": self.version,
                    "file": str(file_path),
                    "metadata": metadata,
                    "size_bytes": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat()
                }
            except Exception as e:
                logger.warning(f"Failed to read prompt {prompt_name}: {e}")
                prompts[prompt_name] = {"name": prompt_name, "error": str(e)}
        
        return prompts
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from prompt file comments."""
        metadata = {
            "description": "",
            "variables": [],
            "model_compatibility": []
        }
        
        # Look for Jinja2 comments with metadata
        # Format: {# description: Extract concepts from text #}
        desc_match = re.search(r"\{#\s*description:\s*(.+?)\s*#\}", content)
        if desc_match:
            metadata["description"] = desc_match.group(1).strip()
        
        # Extract variables used in template
        var_pattern = r"\{\{\s*(\w+)"
        vars_found = set(re.findall(var_pattern, content))
        metadata["variables"] = sorted(vars_found)
        
        return metadata
    
    def get_raw(self, prompt_name: str) -> str:
        """
        Get raw template content without rendering.
        
        Args:
            prompt_name: Name of the prompt
        
        Returns:
            Raw template string
        """
        file_path = self._get_file_path(prompt_name)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt not found: {file_path}")
        
        return file_path.read_text(encoding="utf-8")
    
    def reload(self) -> None:
        """Clear template cache to force reload."""
        self._template_cache.clear()
        self._file_mtimes.clear()
        logger.info("Prompt template cache cleared")
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        Get information about current prompt version.
        
        Returns:
            Version metadata
        """
        available = self.list_available()
        
        return {
            "version": self.version,
            "directory": str(self.version_dir),
            "exists": self.version_dir.exists(),
            "prompt_count": len(available),
            "prompts": list(available.keys()),
            "auto_reload": self.auto_reload,
            "cache_enabled": self.cache_templates
        }


class PromptManager:
    """
    Manage multiple prompt versions and A/B testing.
    
    Provides higher-level management of prompts across versions.
    """
    
    def __init__(self, prompts_dir: str = "./prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._loaders: Dict[str, PromptLoader] = {}
    
    def get_loader(self, version: str) -> PromptLoader:
        """Get or create loader for specific version."""
        if version not in self._loaders:
            self._loaders[version] = PromptLoader(
                prompts_dir=str(self.prompts_dir),
                version=version
            )
        return self._loaders[version]
    
    def list_versions(self) -> List[str]:
        """List all available prompt versions."""
        if not self.prompts_dir.exists():
            return []
        
        versions = []
        for item in self.prompts_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                versions.append(item.name)
        
        return sorted(versions)
    
    def compare_versions(
        self,
        prompt_name: str,
        versions: List[str]
    ) -> Dict[str, str]:
        """
        Compare a prompt across versions.
        
        Args:
            prompt_name: Name of prompt to compare
            versions: List of versions to compare
        
        Returns:
            Dictionary mapping version to prompt content
        """
        comparison = {}
        
        for version in versions:
            try:
                loader = self.get_loader(version)
                content = loader.get_raw(prompt_name)
                comparison[version] = content
            except FileNotFoundError:
                comparison[version] = "[Not found in this version]"
        
        return comparison
