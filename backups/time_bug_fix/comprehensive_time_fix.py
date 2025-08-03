# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
comprehensive_time_fix.py
Comprehensive Fix for Time Calculation Bug Across All Files

This script finds and fixes all instances of the time calculation bug:
OLD BUG: time.total_seconds() / 31536000 * 365
CORRECT: time.total_seconds() / (365.25 * 24 * 3600)

Run this script to fix all legacy files in the repository.
"""

import os
import re
import glob
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeCalculationBugFixer:
    """Comprehensive fixer for time calculation bugs across the codebase."""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize the bug fixer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backups" / "time_bug_fix"
        self.fixed_files: List[str] = []
        self.fixes_applied: Dict[str, List[str]] = {}
        
        # Patterns to identify the time calculation bug
        self.bug_patterns = [
            # Pattern 1: Direct calculation with 31536000
            r'\.total_seconds\(\)\s*/\s*31536000\s*\*\s*365',
            r'\.total_seconds\(\)\s*/\s*31536000\.0?\s*\*\s*365',
            
            # Pattern 2: Lambda functions with the bug
            r'lambda\s+\w+:\s*max\([^)]*\.total_seconds\(\)\s*/\s*31536000[^)]*\)\s*\*\s*365',
            
            # Pattern 3: Apply functions with the bug  
            r'\.apply\(\s*lambda\s+\w+:\s*[^)]*\.total_seconds\(\)\s*/\s*31536000[^)]*\*\s*365[^)]*\)',
            
            # Pattern 4: Variable assignments
            r'\w+\s*=\s*[^=]*\.total_seconds\(\)\s*/\s*31536000\s*\*\s*365',
        ]
        
        # File patterns to search (avoid binary files and directories to ignore)
        self.search_patterns = [
            "**/*.py",
            "**/*.ipynb",  # Jupyter notebooks
        ]
        
        self.ignore_patterns = [
            "**/.*",  # Hidden files/directories
            "**/node_modules/**",
            "**/venv/**", 
            "**/env/**",
            "**/__pycache__/**",
            "**/backups/**",
        ]

    def create_backup(self, file_path: Path) -> None:
        """Create backup of file before modification."""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            relative_path = file_path.relative_to(self.project_root)
            backup_path = self.backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Backup created: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")

    def should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        str_path = str(file_path)
        for pattern in self.ignore_patterns:
            if file_path.match(pattern):
                return True
        return False

    def find_files_with_bug(self) -> List[Path]:
        """Find all files that contain the time calculation bug."""
        files_with_bug = []
        
        for pattern in self.search_patterns:
            for file_path in self.project_root.glob(pattern):
                if self.should_ignore_file(file_path) or not file_path.is_file():
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for any bug pattern
                    for bug_pattern in self.bug_patterns:
                        if re.search(bug_pattern, content, re.IGNORECASE):
                            files_with_bug.append(file_path)
                            break
                            
                except (UnicodeDecodeError, PermissionError) as e:
                    logger.warning(f"Skipping {file_path}: {e}")
                    continue
                    
        return files_with_bug

    def fix_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Fix time calculation bugs in a single file.
        
        Returns:
            Tuple of (success, list_of_fixes_applied)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            content = original_content
            fixes_applied = []
            
            # Fix Pattern 1: Direct calculation
            pattern1 = r'\.total_seconds\(\)\s*/\s*31536000\.?0?\s*\*\s*365\.?0?'
            replacement1 = '.total_seconds() / (365.25 * 24 * 3600)'
            new_content, count1 = re.subn(pattern1, replacement1, content, flags=re.IGNORECASE)
            if count1 > 0:
                content = new_content
                fixes_applied.append(f"Fixed {count1} direct time calculations")
            
            # Fix Pattern 2: Lambda with max/round
            pattern2 = r'lambda\s+(\w+):\s*max\(\s*round\(\s*\1\.total_seconds\(\)\s*/\s*31536000\.?0?\s*,\s*\d+\)\s*,\s*[\d\.e-]+\)\s*\*\s*365\.?0?'
            replacement2 = r'lambda \1: max(\1.total_seconds() / (365.25 * 24 * 3600), 1/(365.25 * 24))'
            new_content, count2 = re.subn(pattern2, replacement2, content, flags=re.IGNORECASE)
            if count2 > 0:
                content = new_content
                fixes_applied.append(f"Fixed {count2} lambda time calculations")
            
            # Fix Pattern 3: Apply with lambda
            pattern3 = r'\.apply\(\s*lambda\s+(\w+):\s*max\(\s*round\(\s*\1\.total_seconds\(\)\s*/\s*31536000\.?0?\s*,\s*\d+\)\s*,\s*[\d\.e-]+\)\s*\*\s*365\.?0?\s*\)'
            replacement3 = r'.apply(lambda \1: max(\1.total_seconds() / (365.25 * 24 * 3600), 1/(365.25 * 24)))'
            new_content, count3 = re.subn(pattern3, replacement3, content, flags=re.IGNORECASE)
            if count3 > 0:
                content = new_content
                fixes_applied.append(f"Fixed {count3} apply time calculations")
            
            # Add import statement if fixes were applied and file is Python
            if fixes_applied and file_path.suffix == '.py' and 'from src.core.utils.time_utils import' not in content:
                # Add import at the top after existing imports
                import_line = "\n# ADDED: Fixed time calculation utilities\nfrom src.core.utils.time_utils import calculate_time_to_maturity_vectorized, fix_legacy_time_calculation\n"
                
                # Find the last import line
                lines = content.split('\n')
                import_end_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                        import_end_idx = i
                
                if import_end_idx > 0:
                    lines.insert(import_end_idx + 1, import_line)
                    content = '\n'.join(lines)
                    fixes_applied.append("Added time utilities import")
            
            # Only write if changes were made
            if content != original_content:
                # Create backup first
                self.create_backup(file_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                return True, fixes_applied
            else:
                return False, []
                
        except Exception as e:
            logger.error(f"Failed to fix {file_path}: {e}")
            return False, [f"Error: {e}"]

    def add_fix_comment(self, file_path: Path) -> None:
        """Add a comment indicating the file was fixed."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            fix_comment = f"""# FIXED: Time calculation bug corrected on {datetime.now().strftime('%Y-%m-%d')}
# OLD BUG: time.total_seconds() / 31536000 * 365  # Mathematically wrong!
# CORRECT: time.total_seconds() / (365.25 * 24 * 3600)  # Proper conversion

"""
            
            # Add comment at the top after the copyright header
            lines = content.split('\n')
            insert_idx = 0
            
            # Find end of copyright/license header
            for i, line in enumerate(lines):
                if line.strip().startswith('#') and ('copyright' in line.lower() or 'license' in line.lower()):
                    continue
                elif line.strip().startswith('#'):
                    continue
                else:
                    insert_idx = i
                    break
            
            lines.insert(insert_idx, fix_comment)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            logger.error(f"Failed to add fix comment to {file_path}: {e}")

    def fix_all_files(self) -> Dict[str, List[str]]:
        """
        Find and fix all files with time calculation bugs.
        
        Returns:
            Dictionary mapping file paths to list of fixes applied
        """
        logger.info("🔍 Scanning for files with time calculation bugs...")
        
        files_with_bug = self.find_files_with_bug()
        
        if not files_with_bug:
            logger.info("✅ No files found with time calculation bugs!")
            return {}
        
        logger.info(f"🚨 Found {len(files_with_bug)} files with time calculation bugs:")
        for file_path in files_with_bug:
            logger.info(f"  - {file_path}")
        
        print(f"\n🔧 Proceeding to fix {len(files_with_bug)} files...")
        input("Press Enter to continue or Ctrl+C to abort...")
        
        results = {}
        fixed_count = 0
        
        for file_path in files_with_bug:
            logger.info(f"Fixing {file_path}...")
            success, fixes = self.fix_file(file_path)
            
            if success:
                self.fixed_files.append(str(file_path))
                results[str(file_path)] = fixes
                fixed_count += 1
                logger.info(f"  ✅ Fixed: {', '.join(fixes)}")
            else:
                results[str(file_path)] = fixes if fixes else ["No changes needed"]
                logger.info(f"  ℹ️ No changes: {', '.join(fixes) if fixes else 'No bugs found'}")
        
        logger.info(f"\n🎉 Fix completed! {fixed_count}/{len(files_with_bug)} files fixed.")
        
        if self.backup_dir.exists():
            logger.info(f"📁 Backups created in: {self.backup_dir}")
        
        return results

    def generate_report(self, results: Dict[str, List[str]]) -> str:
        """Generate a detailed report of fixes applied."""
        from datetime import datetime
        
        report = f"""
# Time Calculation Bug Fix Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Files scanned**: {len(results)} files found with bugs
- **Files fixed**: {len([f for f, fixes in results.items() if any('Fixed' in fix for fix in fixes)])}
- **Backup location**: {self.backup_dir}

## Bug Fixed
**Old (Incorrect)**: `time.total_seconds() / 31536000 * 365`
**New (Correct)**: `time.total_seconds() / (365.25 * 24 * 3600)`

**Why this matters**: The old calculation was mathematically incorrect and would cause
significant errors in options pricing, Greeks calculations, and financial analysis.

## Files Fixed
"""
        
        for file_path, fixes in results.items():
            if any('Fixed' in fix for fix in fixes):
                report += f"\n### {file_path}\n"
                for fix in fixes:
                    report += f"- {fix}\n"
        
        report += "\n## Files Scanned (No fixes needed)\n"
        for file_path, fixes in results.items():
            if not any('Fixed' in fix for fix in fixes):
                report += f"- {file_path}: {', '.join(fixes)}\n"
        
        report += """
## Validation Recommended
After applying these fixes, please:
1. Run the test suite: `python -m pytest tests/test_time_utils.py -v`
2. Validate options pricing accuracy
3. Check that Greeks calculations are working correctly
4. Verify dashboard time displays are accurate

## Rollback Instructions
If needed, restore from backups:
```bash
cp -r backups/time_bug_fix/* .
```
"""
        
        return report


def main():
    """Main function to run the comprehensive time calculation bug fix."""
    import sys
    from datetime import datetime
    
    print("🚀 Qortfolio V2 - Time Calculation Bug Fix")
    print("=" * 50)
    print("This script will find and fix all instances of the time calculation bug:")
    print("OLD BUG: time.total_seconds() / 31536000 * 365")
    print("CORRECT: time.total_seconds() / (365.25 * 24 * 3600)")
    print()
    
    project_root = input("Enter project root directory (or press Enter for current directory): ").strip()
    if not project_root:
        project_root = "."
    
    # Initialize the fixer
    fixer = TimeCalculationBugFixer(project_root)
    
    # Run the fix
    results = fixer.fix_all_files()
    
    # Generate and save report
    report = fixer.generate_report(results)
    report_file = Path(project_root) / "time_bug_fix_report.md"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print("\n✅ Time calculation bug fix completed!")
    
    if results:
        print("\n⚠️ IMPORTANT: Please run tests to validate the fixes:")
        print("python -m pytest tests/test_time_utils.py -v")


if __name__ == "__main__":
    main()