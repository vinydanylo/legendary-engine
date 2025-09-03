import json
import random
import itertools
from typing import List, Tuple, Dict

class CSharpCodeGenerator:
    def __init__(self):
        # Using statements variations
        self.using_statements = [
            "",
            "using System;",
            "using System;\nusing System.Collections.Generic;",
            "using System;\nusing System.Linq;\nusing System.Collections.Generic;",
            "using System;\nusing System.Threading.Tasks;\nusing System.Collections.Generic;\nusing System.Linq;",
            "using System.IO;\nusing System.Text;\nusing System.Net.Http;",
            "using Microsoft.AspNetCore.Mvc;\nusing Microsoft.Extensions.Logging;",
            "using System.Data;\nusing System.Data.SqlClient;\nusing System.Configuration;",
            "using Newtonsoft.Json;\nusing System.ComponentModel.DataAnnotations;",
            "using System.Text.RegularExpressions;\nusing System.Globalization;"
        ]
        
        # Namespace templates
        self.namespace_templates = [
            "",
            "namespace MyApp",
            "namespace MyApp.Services",
            "namespace MyApp.Models",
            "namespace MyApp.Controllers",
            "namespace MyApp.Data.Repositories",
            "namespace Company.Product.Module",
            "namespace Utils.Helpers",
            "namespace Api.v1.Controllers",
            "namespace Domain.Entities"
        ]
        
        # Class templates with different structures
        self.class_templates = [
            # Simple classes
            "public class {name} {{ {content} }}",
            "internal class {name} {{ {content} }}",
            "public static class {name} {{ {content} }}",
            "public abstract class {name} {{ {content} }}",
            "public sealed class {name} {{ {content} }}",
            "public partial class {name} {{ {content} }}",
            
            # Classes with inheritance
            "public class {name} : BaseClass {{ {content} }}",
            "public class {name} : IDisposable {{ {content} }}",
            "public class {name} : Controller {{ {content} }}",
            "public class {name} : Exception {{ {content} }}",
            
            # Generic classes
            "public class {name}<T> {{ {content} }}",
            "public class {name}<T, U> where T : class where U : new() {{ {content} }}",
            "public class {name}<T> : IEnumerable<T> where T : IComparable<T> {{ {content} }}",
            
            # Interfaces and structs
            "public interface I{name} {{ {content} }}",
            "public struct {name} {{ {content} }}",
            "public readonly struct {name} {{ {content} }}",
            "public ref struct {name} {{ {content} }}",
            "public record {name}({content})",
            "public record class {name} {{ {content} }}"
        ]
        
        # Method templates
        self.method_templates = [
            # Basic methods
            "public {return_type} {name}() {{ {body} }}",
            "public {return_type} {name}({params}) {{ {body} }}",
            "private {return_type} {name}({params}) {{ {body} }}",
            "protected {return_type} {name}({params}) {{ {body} }}",
            "internal {return_type} {name}({params}) {{ {body} }}",
            "public static {return_type} {name}({params}) {{ {body} }}",
            "public virtual {return_type} {name}({params}) {{ {body} }}",
            "public override {return_type} {name}({params}) {{ {body} }}",
            "public abstract {return_type} {name}({params});",
            
            # Async methods
            "public async Task<{return_type}> {name}Async() {{ {body} }}",
            "public async Task<{return_type}> {name}Async({params}) {{ {body} }}",
            "public async Task {name}Async({params}) {{ {body} }}",
            
            # Generic methods
            "public T {name}<T>() where T : class {{ {body} }}",
            "public T {name}<T>(T input) where T : IComparable<T> {{ {body} }}",
            "public void {name}<T>(List<T> items) where T : new() {{ {body} }}",
        ]
        
        # Property templates
        self.property_templates = [
            "public {type} {name} {{ get; set; }}",
            "public {type} {name} {{ get; private set; }}",
            "public {type} {name} {{ get; init; }}",
            "public {type} {name} {{ get => {field}; set => {field} = value; }}",
            "public static {type} {name} {{ get; set; }}",
            "public virtual {type} {name} {{ get; set; }}",
            "public override {type} {name} {{ get; set; }}",
            "private {type} {name} {{ get; set; }}",
            "protected {type} {name} {{ get; set; }}",
            "internal {type} {name} {{ get; set; }}"
        ]
        
        # Common C# types
        self.types = [
            "int", "string", "bool", "double", "decimal", "float", "long", "byte",
            "DateTime", "TimeSpan", "Guid", "object", "dynamic",
            "List<int>", "List<string>", "Dictionary<string, int>", "HashSet<string>",
            "IEnumerable<T>", "IList<T>", "IDictionary<string, object>",
            "Task<int>", "Task<string>", "Task<bool>", "Task",
            "Person", "User", "Product", "Order", "Customer", "Employee",
            "HttpClient", "StreamReader", "FileStream", "StringBuilder"
        ]
        
        # Parameter patterns
        self.parameter_patterns = [
            "",
            "int id",
            "string name",
            "int id, string name",
            "string input, bool validate = true",
            "object value, Type targetType",
            "IEnumerable<T> items",
            "CancellationToken cancellationToken = default",
            "int page = 1, int size = 10",
            "string connectionString, TimeSpan timeout",
            "params object[] args",
            "ref int result",
            "out string error",
            "in ReadOnlySpan<char> input"
        ]
        
        # Method body patterns - working versions
        self.working_bodies = [
            "return default;",
            "return {default_value};",
            "return new {type}();",
            "throw new NotImplementedException();",
            "Console.WriteLine(\"Method called\");",
            "await Task.CompletedTask;",
            "return await Task.FromResult({default_value});",
            "if (input == null) throw new ArgumentNullException(nameof(input)); return input;",
            "try {{ return ProcessData(); }} catch (Exception ex) {{ logger.LogError(ex.Message); throw; }}",
            "for (int i = 0; i < 10; i++) {{ Console.WriteLine(i); }}",
            "foreach (var item in items) {{ ProcessItem(item); }}",
            "using (var resource = CreateResource()) {{ return resource.Process(); }}",
            "lock (_lockObject) {{ return _sharedResource; }}",
            "return items?.Where(x => x != null).ToList() ?? new List<T>();",
            "var result = await httpClient.GetAsync(url); return await result.Content.ReadAsStringAsync();"
        ]
        
        # Method body patterns - buggy versions  
        self.buggy_bodies = [
            "return default",  # Missing semicolon
            "return new {type}()",  # Missing semicolon
            "Console.WriteLine(\"Method called\")",  # Missing semicolon
            "await Task.CompletedTask",  # Missing semicolon
            "return await Task.FromResult({default_value})",  # Missing semicolon
            "if (input == null) throw new ArgumentNullException(nameof(input)) return input;",  # Missing semicolon
            "return input.Length.ToString();",  # Null reference potential
            "return array[0];",  # Index out of bounds potential
            "return numerator / denominator;",  # Division by zero potential
            "foreach (var item in items) {{ items.Remove(item); }}",  # Collection modification during iteration
            "return (T)value;",  # Invalid cast potential
            "return items[index];",  # Index out of bounds potential
            "var result = await httpClient.GetAsync(url) return await result.Content.ReadAsStringAsync();",  # Missing semicolon
            "for (int i = 0; i <= array.Length; i++) {{ Console.WriteLine(array[i]); }}",  # Off by one error
            "while (condition) {{ /* no condition update */ }}",  # Infinite loop potential
        ]

        # Field templates
        self.field_templates = [
            "private {type} {name};",
            "private readonly {type} {name};",
            "private static {type} {name};",
            "private const {type} {name} = {value};",
            "public {type} {name} = {value};",
            "protected {type} {name};",
            "internal {type} {name};"
        ]

    def generate_full_file_structure(self, is_working: bool = True) -> str:
        """Generate a complete C# file with using statements, namespace, and classes"""
        parts = []
        
        # Add using statements
        using = random.choice(self.using_statements)
        if using:
            parts.append(using)
            parts.append("")
        
        # Add namespace (optional)
        namespace = random.choice(self.namespace_templates)
        if namespace:
            parts.append(f"{namespace}")
            parts.append("{")
            indent = "    "
        else:
            indent = ""
        
        # Add 1-3 classes/interfaces
        num_classes = random.randint(1, 3)
        for i in range(num_classes):
            class_code = self.generate_class_content(is_working, indent)
            parts.append(class_code)
            if i < num_classes - 1:
                parts.append("")
        
        # Close namespace
        if namespace:
            parts.append("}")
        
        return "\n".join(parts)

    def generate_class_content(self, is_working: bool = True, indent: str = "") -> str:
        """Generate a complete class with methods and properties"""
        class_name = random.choice(["Calculator", "Person", "Product", "Service", "Manager", "Helper", "Processor", "Handler", "Controller", "Repository"])
        template = random.choice(self.class_templates)
        
        content_parts = []
        
        # Add fields (0-3)
        num_fields = random.randint(0, 3)
        for _ in range(num_fields):
            field_type = random.choice(self.types)
            field_name = f"_{random.choice(['data', 'value', 'item', 'result', 'count', 'index'])}"
            field_template = random.choice(self.field_templates)
            field_code = field_template.format(type=field_type, name=field_name, value=self.get_default_value(field_type))
            content_parts.append(f"{indent}    {field_code}")
        
        # Add properties (1-4)
        num_properties = random.randint(1, 4)
        for _ in range(num_properties):
            prop_type = random.choice(self.types)
            prop_name = random.choice(["Id", "Name", "Value", "Count", "Status", "Data", "Result", "Item"])
            prop_template = random.choice(self.property_templates)
            prop_code = prop_template.format(type=prop_type, name=prop_name, field=f"_{prop_name.lower()}")
            if is_working:
                content_parts.append(f"{indent}    {prop_code}")
            else:
                # Introduce syntax errors in properties
                buggy_code = prop_code.replace(";", "") if random.random() > 0.5 else prop_code
                content_parts.append(f"{indent}    {buggy_code}")
        
        # Add methods (1-5)
        num_methods = random.randint(1, 5)
        for _ in range(num_methods):
            method_code = self.generate_method(is_working, f"{indent}    ")
            content_parts.append(method_code)
        
        content = "\n".join(content_parts)
        
        class_code = template.format(name=class_name, content="\n" + content + "\n" + indent)
        return f"{indent}{class_code}"

    def generate_method(self, is_working: bool = True, indent: str = "") -> str:
        """Generate a single method"""
        method_name = random.choice(["Process", "Calculate", "Validate", "Create", "Update", "Delete", "Get", "Set", "Find", "Save", "Load", "Execute"])
        return_type = random.choice(self.types)
        params = random.choice(self.parameter_patterns)
        template = random.choice(self.method_templates)
        
        if is_working:
            body = random.choice(self.working_bodies)
        else:
            body = random.choice(self.buggy_bodies)
        
        # Replace placeholders
        body = body.format(
            default_value=self.get_default_value(return_type),
            type=return_type
        )
        
        method_code = template.format(
            return_type=return_type,
            name=method_name,
            params=params,
            body=body
        )
        
        return f"{indent}{method_code}"

    def generate_single_method(self, is_working: bool = True) -> str:
        """Generate just a single method without class wrapper"""
        return self.generate_method(is_working).strip()

    def generate_single_class(self, is_working: bool = True) -> str:
        """Generate just a single class without namespace or using statements"""
        return self.generate_class_content(is_working).strip()

    def generate_code_snippet(self, is_working: bool = True) -> str:
        """Generate a small code snippet (few lines)"""
        snippet_types = [
            "property", "field", "method_call", "variable_declaration", 
            "loop", "conditional", "try_catch", "using_statement"
        ]
        
        snippet_type = random.choice(snippet_types)
        
        if snippet_type == "property":
            prop_type = random.choice(self.types)
            prop_name = random.choice(["Id", "Name", "Value", "Status"])
            code = f"public {prop_type} {prop_name} {{ get; set; }}"
            if not is_working:
                code = code.replace(";", "")  # Remove semicolon
            return code
            
        elif snippet_type == "field":
            field_type = random.choice(self.types)
            field_name = f"_{random.choice(['data', 'value', 'count'])}"
            code = f"private {field_type} {field_name} = {self.get_default_value(field_type)};"
            if not is_working:
                code = code.replace(";", "")
            return code
            
        elif snippet_type == "method_call":
            method_name = random.choice(["Process", "Calculate", "Validate"])
            code = f"var result = {method_name}(input);"
            if not is_working:
                if random.random() > 0.5:
                    code = code.replace(";", "")
                else:
                    code = f"var result = {method_name}(null);"  # Null input
            return code
            
        elif snippet_type == "variable_declaration":
            var_type = random.choice(self.types)
            var_name = random.choice(["result", "data", "value", "item"])
            code = f"{var_type} {var_name} = {self.get_default_value(var_type)};"
            if not is_working:
                code = code.replace(";", "")
            return code
            
        elif snippet_type == "loop":
            code = "for (int i = 0; i < count; i++) { Console.WriteLine(i); }"
            if not is_working:
                if random.random() > 0.5:
                    code = "for (int i = 0; i <= count; i++) { Console.WriteLine(array[i]); }"  # Off by one
                else:
                    code = code.replace(";", "", 1)  # Remove semicolon
            return code
            
        elif snippet_type == "conditional":
            code = "if (value != null) { ProcessValue(value); }"
            if not is_working:
                if random.random() > 0.5:
                    code = "if (value = null) { ProcessValue(value); }"  # Assignment instead of comparison
                else:
                    code = "ProcessValue(value);"  # No null check
            return code
            
        elif snippet_type == "try_catch":
            code = "try { var result = RiskyOperation(); } catch (Exception ex) { logger.LogError(ex.Message); }"
            if not is_working:
                if random.random() > 0.5:
                    code = code.replace(";", "", 1)  # Remove semicolon
                else:
                    code = "var result = RiskyOperation();"  # No exception handling
            return code
            
        elif snippet_type == "using_statement":
            code = "using (var stream = new FileStream(path, FileMode.Open)) { return stream.ReadByte(); }"
            if not is_working:
                if random.random() > 0.5:
                    code = "var stream = new FileStream(path, FileMode.Open); return stream.ReadByte();"  # No using
                else:
                    code = code.replace(";", "", 1)  # Remove semicolon
            return code
        
        return ""

    def get_default_value(self, type_name: str) -> str:
        """Get a default value for a given type"""
        defaults = {
            "int": "0",
            "string": "\"\"",
            "bool": "false",
            "double": "0.0",
            "decimal": "0m",
            "float": "0f",
            "long": "0L",
            "byte": "0",
            "DateTime": "DateTime.Now",
            "TimeSpan": "TimeSpan.Zero",
            "Guid": "Guid.NewGuid()",
            "object": "null",
            "dynamic": "null"
        }
        
        if type_name in defaults:
            return defaults[type_name]
        elif type_name.startswith("List<"):
            return f"new {type_name}()"
        elif type_name.startswith("Dictionary<"):
            return f"new {type_name}()"
        elif type_name.startswith("Task<"):
            inner_type = type_name[5:-1]
            return f"Task.FromResult({self.get_default_value(inner_type)})"
        elif type_name == "Task":
            return "Task.CompletedTask"
        else:
            return "null"

    def generate_dataset(self, num_samples: int) -> List[Tuple[str, int]]:
        """Generate a balanced dataset with various code structures"""
        dataset = []
        
        # Generate working samples (label 0)
        for i in range(num_samples // 2):
            structure_type = random.choice([
                "full_file", "full_file", "single_class", "single_class", 
                "single_method", "single_method", "code_snippet", "using_only"
            ])
            
            if structure_type == "full_file":
                code = self.generate_full_file_structure(is_working=True)
            elif structure_type == "single_class":
                code = self.generate_single_class(is_working=True)
            elif structure_type == "single_method":
                code = self.generate_single_method(is_working=True)
            elif structure_type == "code_snippet":
                code = self.generate_code_snippet(is_working=True)
            elif structure_type == "using_only":
                code = random.choice(self.using_statements) + "\n\npublic class Program { static void Main() { Console.WriteLine(\"Hello\"); } }"
            
            dataset.append((code, 0))
        
        # Generate buggy samples (label 1)
        for i in range(num_samples // 2):
            structure_type = random.choice([
                "full_file", "full_file", "single_class", "single_class",
                "single_method", "single_method", "code_snippet", "using_only"
            ])
            
            if structure_type == "full_file":
                code = self.generate_full_file_structure(is_working=False)
            elif structure_type == "single_class":
                code = self.generate_single_class(is_working=False)
            elif structure_type == "single_method":
                code = self.generate_single_method(is_working=False)
            elif structure_type == "code_snippet":
                code = self.generate_code_snippet(is_working=False)
            elif structure_type == "using_only":
                code = random.choice(self.using_statements).replace(";", "") + "\n\npublic class Program { static void Main() { Console.WriteLine(\"Hello\"); } }"
            
            dataset.append((code, 1))
        
        # Handle odd number
        if num_samples % 2 == 1:
            code = self.generate_single_method(is_working=True)
            dataset.append((code, 0))
        
        return dataset

def create_enhanced_dataset(filename: str, num_samples: int):
    """Create an enhanced JSONL dataset with varied C# code structures"""
    generator = CSharpCodeGenerator()
    dataset = generator.generate_dataset(num_samples)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        for code, label in dataset:
            sample = {"code": code.strip(), "label": label}
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Created {filename} with {len(dataset)} samples")
    print(f"Working samples: {sum(1 for _, label in dataset if label == 0)}")
    print(f"Buggy samples: {sum(1 for _, label in dataset if label == 1)}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Generating enhanced C# dataset with varied structures...")
    
    # Create training dataset (50,000 samples)
    create_enhanced_dataset("train_csharp_bugs_enhanced.jsonl", 50000)
    
    # Create validation dataset (10,000 samples)  
    create_enhanced_dataset("valid_csharp_bugs_enhanced.jsonl", 10000)
    
    print("Enhanced dataset generation complete!")
    print("\nDataset includes:")
    print("- Complete files with using statements and namespaces")
    print("- Individual classes with various inheritance patterns")
    print("- Single methods with different signatures")
    print("- Code snippets (properties, fields, loops, etc.)")
    print("- Generic classes and methods")
    print("- Async/await patterns")
    print("- LINQ expressions")
    print("- Various C# language features")