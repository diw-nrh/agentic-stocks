import ast

def verify_skill_syntax(source_code: str) -> dict:
    """
    Self-RAG / CRAG Verification Component:
    Checks if the generated or retrieved skill (Python code) is valid and safe.
    """
    try:
        # Check 1: Syntax Parsing
        tree = ast.parse(source_code)
        
        # Check 2: Verify it contains a @tool decorator (basic check)
        has_tool_decorator = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'tool':
                        has_tool_decorator = True
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'tool':
                        has_tool_decorator = True
                        
        if not has_tool_decorator:
            return {
                "status": "failed",
                "reason": "The source code does not contain a function decorated with @tool."
            }
            
        return {
            "status": "passed",
            "reason": "Syntax is valid and @tool decorator is present."
        }
    except SyntaxError as e:
        return {
            "status": "failed",
            "reason": f"Syntax Error in source code: {e}"
        }
    except Exception as e:
        return {
            "status": "failed",
            "reason": f"Unknown Error during verification: {e}"
        }
