def get_skill_description(skill_path: str) -> str:
    """
    function for reading the skill description from a .md file
    """
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading skill description: {e}"