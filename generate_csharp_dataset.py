import json
import random

def generate_working_code_samples():
    """Generate syntactically correct and functional C# code samples"""
    return [
        'public class Calculator { public int Add(int a, int b) { return a + b; } }',
        'using System; public class Program { static void Main() { Console.WriteLine("Hello World"); } }',
        'public int Multiply(int x, int y) { return x * y; }',
        'public class Person { public string Name { get; set; } public int Age { get; set; } }',
        'public List<int> numbers = new List<int> { 1, 2, 3, 4, 5 };',
        'public void PrintNumbers() { for (int i = 0; i < 10; i++) { Console.WriteLine(i); } }',
        'public string GetFullName(string first, string last) { return $"{first} {last}"; }',
        'public class Rectangle { private double width, height; public double Area() { return width * height; } }',
        'public async Task<string> GetDataAsync() { await Task.Delay(1000); return "data"; }',
        'public bool IsEven(int number) { return number % 2 == 0; }',
        'public Dictionary<string, int> scores = new Dictionary<string, int>();',
        'public void ProcessArray(int[] arr) { Array.Sort(arr); }',
        'public class Student : Person { public string StudentId { get; set; } }',
        'public enum Status { Active, Inactive, Pending }',
        'public interface IRepository { void Save(object entity); }',
        'public void TryParseExample() { try { int result = 10 / 5; } catch (Exception ex) { Console.WriteLine(ex.Message); } }',
        'public void ValidateInput(string input) { if (string.IsNullOrEmpty(input)) throw new ArgumentException("Input cannot be null"); }',
        'public class DatabaseConnection : IDisposable { public void Dispose() { /* cleanup */ } }',
        'public T FindById<T>(int id) where T : class { return default(T); }',
        'public void ConfigureServices(IServiceCollection services) { services.AddSingleton<ILogger, ConsoleLogger>(); }',
        'public decimal CalculatePrice(decimal basePrice, decimal taxRate) { return basePrice * (1 + taxRate); }',
        'public void WriteToFile(string path, string content) { File.WriteAllText(path, content); }',
        'public class ApiController : ControllerBase { [HttpGet] public IActionResult Get() { return Ok("Success"); } }',
        'public void UpdateStatus(int id, Status status) { var entity = context.Find(id); entity.Status = status; context.SaveChanges(); }',
        'public string FormatDate(DateTime date) { return date.ToString("yyyy-MM-dd"); }',
        'public List<T> Filter<T>(List<T> items, Func<T, bool> predicate) { return items.Where(predicate).ToList(); }',
        'public void LogMessage(string message) { logger.LogInformation($"[{DateTime.Now}] {message}"); }',
        'public class WeatherService { public async Task<Weather> GetWeatherAsync(string city) { return await httpClient.GetFromJsonAsync<Weather>($"api/weather/{city}"); } }',
        'public void SendEmail(string to, string subject, string body) { var message = new MailMessage(from, to, subject, body); smtpClient.Send(message); }',
        'public string HashPassword(string password) { return BCrypt.Net.BCrypt.HashPassword(password); }',
        'public abstract class Shape { public abstract double GetArea(); public virtual void Draw() { Console.WriteLine("Drawing shape"); } }',
        'public class Circle : Shape { private double radius; public Circle(double r) { radius = r; } public override double GetArea() { return Math.PI * radius * radius; } }',
        'public delegate void EventHandler<T>(object sender, T eventArgs); public event EventHandler<string> StatusChanged;',
        'public class GenericRepository<T> : IRepository<T> where T : class, new() { public void Add(T entity) { context.Set<T>().Add(entity); } }',
        'public async Task<IEnumerable<T>> GetAllAsync<T>() where T : class { return await context.Set<T>().ToListAsync(); }',
        'public class ThreadSafeCounter { private readonly object _lock = new object(); private int _count = 0; public void Increment() { lock (_lock) { _count++; } } }',
        'public record PersonRecord(string FirstName, string LastName, int Age) { public string FullName => $"{FirstName} {LastName}"; }',
        'public class LazyInitialization { private readonly Lazy<ExpensiveObject> _expensive = new Lazy<ExpensiveObject>(() => new ExpensiveObject()); }',
        'public void UseUsing() { using (var connection = new SqlConnection(connectionString)) { connection.Open(); /* use connection */ } }',
        'public class Observer : INotifyPropertyChanged { public event PropertyChangedEventHandler PropertyChanged; protected void OnPropertyChanged([CallerMemberName] string name = null) { PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name)); } }',
        'public static class Extensions { public static bool IsNullOrEmpty(this string value) { return string.IsNullOrEmpty(value); } }',
        'public class Builder { private string _name; public Builder SetName(string name) { _name = name; return this; } public Person Build() { return new Person { Name = _name }; } }',
        'public class Singleton { private static readonly Lazy<Singleton> _instance = new Lazy<Singleton>(() => new Singleton()); public static Singleton Instance => _instance.Value; }',
        'public void PatternMatching(object obj) { if (obj is string s && s.Length > 0) { Console.WriteLine($"String: {s}"); } }',
        'public string SwitchExpression(int value) => value switch { 1 => "One", 2 => "Two", 3 => "Three", _ => "Other" };',
        'public class NullableExample { public string? Name { get; set; } public void ProcessName() { if (Name is not null) { Console.WriteLine(Name.ToUpper()); } } }',
        'public void DeconstructExample() { var (x, y) = GetCoordinates(); Console.WriteLine($"X: {x}, Y: {y}"); } private (int, int) GetCoordinates() => (10, 20);',
        'public class AsyncEnumerableExample { public async IAsyncEnumerable<int> GetNumbersAsync() { for (int i = 0; i < 10; i++) { await Task.Delay(100); yield return i; } } }',
        'public void LocalFunction() { int Add(int a, int b) => a + b; var result = Add(5, 3); Console.WriteLine(result); }',
        'public class IndexAndRange { public void Example() { var array = new int[] { 1, 2, 3, 4, 5 }; var slice = array[1..4]; var last = array[^1]; } }'
    ]

def generate_buggy_code_samples():
    """Generate C# code with various types of bugs"""
    return [
        'public class Calculator { public int Add(int a, int b) { return a + b }',  # Missing semicolon
        'using System public class Program { static void Main() { Console.WriteLine("Hello World"); } }',  # Missing semicolon after using
        'public int Multiply(int x, int y) { return x * y',  # Missing closing brace and semicolon
        'public class Person { public string Name { get; set } public int Age { get; set; } }',  # Missing semicolon after get; set
        'public List<int> numbers = new List<int> { 1, 2, 3, 4, 5 }',  # Missing semicolon
        'public void PrintNumbers() { for (int i = 0; i < 10; i++) Console.WriteLine(i); } }',  # Extra closing brace
        'public string GetFullName(string first, string last) { return first + " " + last }',  # Missing semicolon
        'public class Rectangle { private double width, height; public double Area() { return width * height } }',  # Missing semicolon
        'public async Task<string> GetDataAsync() { await Task.Delay(1000) return "data"; }',  # Missing semicolon
        'public bool IsEven(int number) { return number % 2 == 0 }',  # Missing semicolon
        'public Dictionary<string, int> scores = new Dictionary<string, int>()',  # Missing semicolon
        'public void ProcessArray(int[] arr) { Array.Sort(arr) }',  # Missing semicolon
        'public class Student : Person { public string StudentId { get set; } }',  # Missing semicolon after get
        'public enum Status { Active, Inactive, Pending',  # Missing closing brace
        'public interface IRepository { void Save(object entity) }',  # Missing semicolon
        'public void TryParseExample() { try { int result = 10 / 5 } catch (Exception ex) { Console.WriteLine(ex.Message); } }',  # Missing semicolon in try block
        'public void ValidateInput(string input) { if (string.IsNullOrEmpty(input) throw new ArgumentException("Input cannot be null"); }',  # Missing closing parenthesis
        'public class DatabaseConnection : IDisposable { public void Dispose() { /* cleanup */ }',  # Missing closing brace
        'public T FindById<T>(int id) where T : class { return default(T) }',  # Missing semicolon
        'public void ConfigureServices(IServiceCollection services) { services.AddSingleton<ILogger, ConsoleLogger>() }',  # Missing semicolon
        'public decimal CalculatePrice(decimal basePrice, decimal taxRate) { return basePrice * (1 + taxRate }',  # Missing closing parenthesis and semicolon
        'public void WriteToFile(string path, string content) { File.WriteAllText(path, content) }',  # Missing semicolon
        'public class ApiController : ControllerBase { [HttpGet] public IActionResult Get() { return Ok("Success") } }',  # Missing semicolon
        'public void UpdateStatus(int id, Status status) { var entity = context.Find(id); entity.Status = status context.SaveChanges(); }',  # Missing semicolon
        'public string FormatDate(DateTime date) { return date.ToString("yyyy-MM-dd") }',  # Missing semicolon
        'public List<T> Filter<T>(List<T> items, Func<T, bool> predicate) { return items.Where(predicate).ToList() }',  # Missing semicolon
        'public void LogMessage(string message) { logger.LogInformation($"[{DateTime.Now}] {message}") }',  # Missing semicolon
        'public class WeatherService { public async Task<Weather> GetWeatherAsync(string city) { return await httpClient.GetFromJsonAsync<Weather>($"api/weather/{city}") } }',  # Missing semicolon
        'public void SendEmail(string to, string subject, string body) { var message = new MailMessage(from, to, subject, body) smtpClient.Send(message); }',  # Missing semicolon
        'public string HashPassword(string password) { return BCrypt.Net.BCrypt.HashPassword(password) }',  # Missing semicolon
        'public int Divide(int a, int b) { return a / b; }',  # Division by zero potential bug
        'public void AccessArray(int[] arr, int index) { return arr[index]; }',  # Array index out of bounds potential
        'public string GetSubstring(string text, int start) { return text.Substring(start); }',  # Potential null reference
        'public void CloseFile(FileStream file) { file.Close(); }',  # Potential null reference
        'public int ParseNumber(string input) { return int.Parse(input); }',  # Parse exception potential
        'public void ProcessList(List<string> items) { foreach (var item in items) { items.Remove(item); } }',  # Modifying collection during iteration
        'public async void SaveData() { await database.SaveAsync(); }',  # async void instead of async Task
        'public class Singleton { private static Singleton instance; public static Singleton Instance { get { return instance ?? new Singleton(); } } }',  # Thread safety issue
        'public void HandleEvent() { SomeEvent += (s, e) => { throw new Exception("Error"); }; }',  # Unhandled exception in event handler
        'public string ConcatenateStrings(string[] strings) { string result = ""; foreach (string s in strings) { result += s; } return result; }',  # Inefficient string concatenation
        'public abstract class Shape { public abstract double GetArea() public virtual void Draw() { Console.WriteLine("Drawing shape"); } }',  # Missing semicolon
        'public class Circle : Shape { private double radius; public Circle(double r) { radius = r } public override double GetArea() { return Math.PI * radius * radius; } }',  # Missing semicolon
        'public delegate void EventHandler<T>(object sender, T eventArgs) public event EventHandler<string> StatusChanged;',  # Missing semicolon
        'public class GenericRepository<T> : IRepository<T> where T : class, new() { public void Add(T entity) { context.Set<T>().Add(entity) } }',  # Missing semicolon
        'public async Task<IEnumerable<T>> GetAllAsync<T>() where T : class { return await context.Set<T>().ToListAsync() }',  # Missing semicolon
        'public class ThreadSafeCounter { private readonly object _lock = new object(); private int _count = 0; public void Increment() { lock (_lock) { _count++ } } }',  # Missing semicolon
        'public record PersonRecord(string FirstName, string LastName, int Age) { public string FullName => $"{FirstName} {LastName}" }',  # Missing semicolon
        'public class LazyInitialization { private readonly Lazy<ExpensiveObject> _expensive = new Lazy<ExpensiveObject>(() => new ExpensiveObject()) }',  # Missing semicolon
        'public void UseUsing() { using (var connection = new SqlConnection(connectionString)) { connection.Open(); /* use connection */ }',  # Missing closing brace
        'public class Observer : INotifyPropertyChanged { public event PropertyChangedEventHandler PropertyChanged; protected void OnPropertyChanged([CallerMemberName] string name = null) { PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name)) } }',  # Missing semicolon
        'public static class Extensions { public static bool IsNullOrEmpty(this string value) { return string.IsNullOrEmpty(value) } }',  # Missing semicolon
        'public class Builder { private string _name; public Builder SetName(string name) { _name = name; return this } public Person Build() { return new Person { Name = _name }; } }',  # Missing semicolon
        'public class Singleton { private static readonly Lazy<Singleton> _instance = new Lazy<Singleton>(() => new Singleton()) public static Singleton Instance => _instance.Value; }',  # Missing semicolon
        'public void PatternMatching(object obj) { if (obj is string s && s.Length > 0) { Console.WriteLine($"String: {s}") } }',  # Missing semicolon
        'public string SwitchExpression(int value) => value switch { 1 => "One", 2 => "Two", 3 => "Three", _ => "Other" }',  # Missing semicolon
        'public class NullableExample { public string? Name { get; set; } public void ProcessName() { if (Name is not null) { Console.WriteLine(Name.ToUpper()) } } }',  # Missing semicolon
        'public void DeconstructExample() { var (x, y) = GetCoordinates(); Console.WriteLine($"X: {x}, Y: {y}") } private (int, int) GetCoordinates() => (10, 20);',  # Missing semicolon
        'public class AsyncEnumerableExample { public async IAsyncEnumerable<int> GetNumbersAsync() { for (int i = 0; i < 10; i++) { await Task.Delay(100); yield return i } } }',  # Missing semicolon
        'public void LocalFunction() { int Add(int a, int b) => a + b var result = Add(5, 3); Console.WriteLine(result); }',  # Missing semicolon
        'public class IndexAndRange { public void Example() { var array = new int[] { 1, 2, 3, 4, 5 }; var slice = array[1..4] var last = array[^1]; } }',  # Missing semicolon
        'public interface IGeneric<T> where T : class { T GetItem(int id) }',  # Missing semicolon
        'public class ComplexGeneric<T, U> where T : IComparable<T> where U : new() { public void Process(T item, U other) { /* logic */ }',  # Missing closing brace
        'public enum ComplexEnum : int { First = 1, Second = 2, Third = 3',  # Missing closing brace
        'public struct Point { public int X { get; set; } public int Y { get; set; } public Point(int x, int y) { X = x; Y = y }',  # Missing closing brace
        'public class EventExample { public event Action<string> OnDataReceived; public void TriggerEvent(string data) { OnDataReceived?.Invoke(data) } }',  # Missing semicolon
        'public void UnsafeCode() { unsafe { int* ptr = stackalloc int[10]; *ptr = 42 } }',  # Missing semicolon
        'public class OperatorOverload { public static OperatorOverload operator +(OperatorOverload a, OperatorOverload b) { return new OperatorOverload() } }',  # Missing semicolon
        'public partial class PartialClass { partial void OnSomething() }',  # Missing semicolon and implementation
        'public sealed class SealedClass : BaseClass { public override void Method() { base.Method() } }',  # Missing semicolon
        'public ref struct RefStruct { public readonly int Value public RefStruct(int value) { Value = value; } }',  # Missing semicolon
        'public readonly struct ReadonlyStruct { public readonly int Value; public ReadonlyStruct(int value) { Value = value }',  # Missing closing brace
        'public void RefReturn() { ref int GetRef() => ref array[0] var reference = ref GetRef(); }',  # Missing semicolon
        'public void LocalFunctionCapture() { int local = 5; void Inner() { Console.WriteLine(local) } Inner(); }',  # Missing semicolon
    ]

def generate_additional_working_samples():
    """Generate more varied working C# code samples"""
    samples = []
    
    # Complex LINQ queries
    samples.extend([
        'public var complexQuery = from p in people join c in companies on p.CompanyId equals c.Id where p.Age > 25 select new { p.Name, c.CompanyName, p.Salary };',
        'public IEnumerable<T> GetPagedResults<T>(IQueryable<T> query, int page, int size) { return query.Skip((page - 1) * size).Take(size); }',
        'public Dictionary<TKey, TValue[]> GroupByKey<TKey, TValue>(IEnumerable<TValue> items, Func<TValue, TKey> keySelector) { return items.GroupBy(keySelector).ToDictionary(g => g.Key, g => g.ToArray()); }',
        'public async Task<List<T>> ProcessBatchAsync<T>(IEnumerable<T> items, Func<T, Task<T>> processor, int batchSize = 10) { var batches = items.Chunk(batchSize); var results = new List<T>(); foreach (var batch in batches) { var tasks = batch.Select(processor); results.AddRange(await Task.WhenAll(tasks)); } return results; }'
    ])
    
    # Advanced generics and constraints
    samples.extend([
        'public class Repository<TEntity, TKey> : IRepository<TEntity, TKey> where TEntity : class, IEntity<TKey> where TKey : IEquatable<TKey> { public async Task<TEntity> GetByIdAsync(TKey id) { return await context.Set<TEntity>().FindAsync(id); } }',
        'public TResult Map<TSource, TResult>(TSource source, Func<TSource, TResult> mapper) where TSource : class where TResult : class, new() { return source != null ? mapper(source) : new TResult(); }',
        'public class CacheManager<TKey, TValue> where TKey : notnull where TValue : class { private readonly ConcurrentDictionary<TKey, TValue> cache = new(); public TValue GetOrAdd(TKey key, Func<TKey, TValue> factory) { return cache.GetOrAdd(key, factory); } }'
    ])
    
    # Complex async patterns
    samples.extend([
        'public async Task<T> RetryAsync<T>(Func<Task<T>> operation, int maxRetries = 3, TimeSpan? delay = null) { for (int i = 0; i < maxRetries; i++) { try { return await operation(); } catch (Exception) when (i < maxRetries - 1) { if (delay.HasValue) await Task.Delay(delay.Value); } } throw new InvalidOperationException("Max retries exceeded"); }',
        'public async IAsyncEnumerable<T> ProcessStreamAsync<T>(IAsyncEnumerable<T> source, Func<T, Task<T>> processor, [EnumeratorCancellation] CancellationToken ct = default) { await foreach (var item in source.WithCancellation(ct)) { yield return await processor(item); } }',
        'public class AsyncLazy<T> { private readonly Lazy<Task<T>> lazy; public AsyncLazy(Func<Task<T>> taskFactory) { lazy = new Lazy<Task<T>>(taskFactory); } public TaskAwaiter<T> GetAwaiter() => lazy.Value.GetAwaiter(); }'
    ])
    
    # Design patterns
    samples.extend([
        'public class Command : ICommand { private readonly Action action; private readonly Action undoAction; public Command(Action action, Action undoAction) { this.action = action; this.undoAction = undoAction; } public void Execute() => action(); public void Undo() => undoAction(); }',
        'public class Observer<T> : IObserver<T> { private readonly Action<T> onNext; public Observer(Action<T> onNext) { this.onNext = onNext; } public void OnNext(T value) => onNext(value); public void OnError(Exception error) => Console.WriteLine($"Error: {error}"); public void OnCompleted() => Console.WriteLine("Completed"); }',
        'public abstract class TemplateMethod { public void Execute() { Initialize(); ProcessData(); Cleanup(); } protected abstract void ProcessData(); protected virtual void Initialize() { } protected virtual void Cleanup() { } }'
    ])
    
    # Memory management and performance
    samples.extend([
        'public ref struct SpanProcessor<T> where T : unmanaged { private readonly Span<T> span; public SpanProcessor(Span<T> span) { this.span = span; } public void Process(Func<T, T> processor) { for (int i = 0; i < span.Length; i++) { span[i] = processor(span[i]); } } }',
        'public class ObjectPool<T> where T : class, new() { private readonly ConcurrentQueue<T> objects = new(); public T Get() => objects.TryDequeue(out var item) ? item : new T(); public void Return(T item) => objects.Enqueue(item); }',
        'public unsafe void ProcessBuffer(byte* buffer, int length) { for (int i = 0; i < length; i++) { buffer[i] = (byte)(buffer[i] ^ 0xFF); } }'
    ])
    
    # Expression trees and dynamic
    samples.extend([
        'public static Expression<Func<T, bool>> CombinePredicates<T>(Expression<Func<T, bool>> first, Expression<Func<T, bool>> second) { var parameter = Expression.Parameter(typeof(T)); var combined = Expression.AndAlso(Expression.Invoke(first, parameter), Expression.Invoke(second, parameter)); return Expression.Lambda<Func<T, bool>>(combined, parameter); }',
        'public class DynamicPropertyAccessor { public static object GetValue(object obj, string propertyName) { return obj.GetType().GetProperty(propertyName)?.GetValue(obj); } public static void SetValue(object obj, string propertyName, object value) { obj.GetType().GetProperty(propertyName)?.SetValue(obj, value); } }',
        'public T CreateInstance<T>(params object[] args) where T : class { return (T)Activator.CreateInstance(typeof(T), args); }'
    ])
    
    # Advanced C# features
    samples.extend([
        'public readonly record struct ValueRecord(int Id, string Name) { public static implicit operator ValueRecord((int Id, string Name) tuple) => new(tuple.Id, tuple.Name); }',
        'public static class PatternMatching { public static string Describe(object obj) => obj switch { string s when s.Length > 10 => $"Long string: {s[..10]}...", string s => $"Short string: {s}", int i when i > 100 => "Large number", int i => $"Small number: {i}", _ => "Unknown type" }; }',
        'public class DiscriminatedUnion<T1, T2, T3> { private readonly object value; private readonly int tag; public static DiscriminatedUnion<T1, T2, T3> Create1(T1 value) => new() { value = value, tag = 1 }; public TResult Match<TResult>(Func<T1, TResult> f1, Func<T2, TResult> f2, Func<T3, TResult> f3) => tag switch { 1 => f1((T1)value), 2 => f2((T2)value), 3 => f3((T3)value), _ => throw new InvalidOperationException() }; }'
    ])
    
    # Simple methods with variations
    for i in range(30):
        samples.append(f'public int Method{i}() {{ return {i * 10}; }}')
    
    # Properties with different access modifiers
    for i in range(20):
        samples.append(f'public string Property{i} {{ get; set; }} = "value{i}";')
    
    # Classes with inheritance
    for i in range(20):
        samples.append(f'public class Class{i} : BaseClass{{ public int Id {{ get; set; }} = {i}; }}')
    
    # Loop variations
    for i in range(15):
        samples.append(f'public void Loop{i}() {{ for (int j = 0; j < {i + 5}; j++) {{ Console.WriteLine(j); }} }}')
    
    # Conditional variations
    for i in range(15):
        samples.append(f'public bool Check{i}(int value) {{ return value > {i}; }}')
    
    return samples

def generate_additional_buggy_samples():
    """Generate more varied buggy C# code samples"""
    samples = []
    
    # Runtime bugs - null reference exceptions
    samples.extend([
        'public string GetLength(string input) { return input.Length.ToString(); }',  # Null reference exception
        'public void ProcessArray(int[] array) { Console.WriteLine(array[0]); }',  # Null reference exception
        'public void AccessProperty(Person person) { Console.WriteLine(person.Name.ToUpper()); }',  # Null reference exception
        'public void ProcessList(List<string> items) { foreach (var item in items) { Console.WriteLine(item.Length); } }',  # Null reference exception
        'public string ConcatNames(Person person) { return person.FirstName + " " + person.LastName; }',  # Null reference exception
    ])
    
    # Runtime bugs - index out of bounds
    samples.extend([
        'public int GetFirstElement(int[] array) { return array[0]; }',  # Index out of bounds on empty array
        'public char GetCharAt(string text, int index) { return text[index]; }',  # Index out of bounds
        'public T GetItemAt<T>(List<T> list, int index) { return list[index]; }',  # Index out of bounds
        'public void AccessLastElement(int[] numbers) { Console.WriteLine(numbers[numbers.Length]); }',  # Off by one error
        'public string GetSubstring(string text, int start, int length) { return text.Substring(start, length); }',  # Substring out of bounds
    ])
    
    # Runtime bugs - division and arithmetic errors
    samples.extend([
        'public double Divide(int a, int b) { return a / b; }',  # Division by zero
        'public int CalculatePercentage(int value, int total) { return (value * 100) / total; }',  # Division by zero
        'public double CalculateAverage(int[] numbers) { return numbers.Sum() / numbers.Length; }',  # Division by zero with empty array
        'public int ModuloOperation(int a, int b) { return a % b; }',  # Modulo by zero
        'public decimal ParseAndDivide(string numerator, string denominator) { return decimal.Parse(numerator) / decimal.Parse(denominator); }',  # Parse and division errors
    ])
    
    # Runtime bugs - collection modification during iteration
    samples.extend([
        'public void RemoveEvens(List<int> numbers) { foreach (var num in numbers) { if (num % 2 == 0) numbers.Remove(num); } }',  # Collection modified during iteration
        'public void ProcessAndRemove(Dictionary<string, int> dict) { foreach (var kvp in dict) { if (kvp.Value < 0) dict.Remove(kvp.Key); } }',  # Collection modified during iteration
        'public void ClearWhileIterating(List<string> items) { foreach (var item in items) { if (item.Length > 10) items.Clear(); } }',  # Collection modified during iteration
        'public void AddWhileIterating(HashSet<int> numbers) { foreach (var num in numbers) { if (num > 5) numbers.Add(num * 2); } }',  # Collection modified during iteration
        'public void RemoveFromQueue(Queue<string> queue) { foreach (var item in queue) { if (item.StartsWith("temp")) queue.Dequeue(); } }',  # Collection modified during iteration
    ])
    
    # Runtime bugs - type casting and conversion errors
    samples.extend([
        'public int CastToInt(object value) { return (int)value; }',  # Invalid cast exception
        'public string ConvertToString(object obj) { return (string)obj; }',  # Invalid cast exception
        'public T CastGeneric<T>(object value) { return (T)value; }',  # Invalid cast exception
        'public double ParseDouble(string text) { return double.Parse(text); }',  # Format exception
        'public DateTime ParseDate(string dateString) { return DateTime.Parse(dateString); }',  # Format exception
    ])
    
    # Runtime bugs - resource disposal and memory leaks
    samples.extend([
        'public string ReadFileContent(string path) { var stream = new FileStream(path, FileMode.Open); var reader = new StreamReader(stream); return reader.ReadToEnd(); }',  # Resource not disposed
        'public void WriteToFile(string path, string content) { var writer = new StreamWriter(path); writer.Write(content); }',  # Resource not disposed
        'public Bitmap LoadImage(string path) { return new Bitmap(path); }',  # Resource not disposed
        'public void DatabaseQuery() { var connection = new SqlConnection(connectionString); connection.Open(); var command = connection.CreateCommand(); command.ExecuteNonQuery(); }',  # Connection not closed
        'public void ProcessHttpRequest() { var client = new HttpClient(); var response = client.GetAsync("http://api.example.com").Result; }',  # Client not disposed
    ])
    
    # Runtime bugs - async/await misuse and deadlocks
    samples.extend([
        'public string GetDataSync() { return GetDataAsync().Result; }',  # Potential deadlock
        'public void ProcessData() { var task = ProcessDataAsync(); task.Wait(); }',  # Potential deadlock
        'public async Task<string> BadAsyncMethod() { var result = GetDataAsync().Result; return result; }',  # Mixing sync and async
        'public void FireAndForget() { ProcessDataAsync(); }',  # Fire and forget without proper handling
        'public async Task<int> DivideAsync(int a, int b) { return a / b; }',  # Division by zero in async method
    ])
    
    # Runtime bugs - race conditions and threading issues
    samples.extend([
        'public class Counter { private int count = 0; public void Increment() { count++; } public int GetCount() { return count; } }',  # Race condition
        'public class SharedResource { private List<string> items = new List<string>(); public void AddItem(string item) { items.Add(item); } public string GetItem(int index) { return items[index]; } }',  # Thread safety issue
        'public static Dictionary<string, int> Cache = new Dictionary<string, int>(); public void UpdateCache(string key, int value) { Cache[key] = value; }',  # Static field race condition
        'public class LazyLoader { private object data = null; public object GetData() { if (data == null) { data = LoadExpensiveData(); } return data; } }',  # Race condition in lazy loading
        'public void ProcessInParallel(List<int> numbers) { Parallel.ForEach(numbers, num => { numbers.Add(num * 2); }); }',  # Collection modification in parallel
    ])
    
    # Runtime bugs - infinite loops and stack overflow
    samples.extend([
        'public void InfiniteLoop() { while (true) { Console.WriteLine("Running..."); } }',  # Infinite loop
        'public int Fibonacci(int n) { return Fibonacci(n - 1) + Fibonacci(n - 2); }',  # Stack overflow - no base case
        'public void RecursiveCall() { RecursiveCall(); }',  # Infinite recursion
        'public int BadRecursion(int n) { if (n > 0) return BadRecursion(n + 1); return 0; }',  # Stack overflow - wrong direction
        'public void WhileLoop() { int i = 10; while (i > 0) { Console.WriteLine(i); i++; } }',  # Infinite loop - wrong increment
    ])
    
    # Runtime bugs - logic errors and off-by-one
    samples.extend([
        'public bool IsInRange(int value, int min, int max) { return value >= min && value >= max; }',  # Logic error
        'public void PrintNumbers(int count) { for (int i = 1; i <= count; i++) { Console.WriteLine(i); } }',  # Off by one - should start at 0
        'public int[] CopyArray(int[] source) { int[] result = new int[source.Length]; for (int i = 0; i <= source.Length; i++) { result[i] = source[i]; } return result; }',  # Off by one
        'public bool ContainsValue(int[] array, int value) { for (int i = 0; i < array.Length - 1; i++) { if (array[i] == value) return true; } return false; }',  # Missing last element
        'public string ReverseString(string input) { string result = ""; for (int i = input.Length; i >= 0; i--) { result += input[i]; } return result; }',  # Off by one
    ])
    
    # Runtime bugs - performance and memory issues
    samples.extend([
        'public string BuildString(string[] words) { string result = ""; foreach (string word in words) { result += word + " "; } return result; }',  # Inefficient string concatenation
        'public List<int> GetLargeNumbers() { var result = new List<int>(); for (int i = 0; i < 1000000; i++) { result.Add(i * i); } return result; }',  # Memory intensive
        'public void ProcessData(List<string> data) { for (int i = 0; i < data.Count; i++) { for (int j = 0; j < data.Count; j++) { Console.WriteLine($"{data[i]}-{data[j]}"); } } }',  # O(n²) complexity
        'public bool CheckDuplicate(int[] array, int value) { for (int i = 0; i < array.Length; i++) { for (int j = 0; j < array.Length; j++) { if (i != j && array[i] == array[j]) return true; } } return false; }',  # Inefficient algorithm
        'public void LoadAllData() { var allRecords = database.GetAllRecords().ToList(); foreach (var record in allRecords) { ProcessRecord(record); } }',  # Loading too much data
    ])
    
    return samples

def create_dataset(filename, num_samples):
    """Create a JSONL dataset with the specified number of samples"""
    working_samples = generate_working_code_samples() + generate_additional_working_samples()
    buggy_samples = generate_buggy_code_samples() + generate_additional_buggy_samples()
    
    # Ensure we have enough samples
    while len(working_samples) < num_samples // 2:
        working_samples.extend(generate_additional_working_samples())
    
    while len(buggy_samples) < num_samples // 2:
        buggy_samples.extend(generate_additional_buggy_samples())
    
    # Create balanced dataset
    dataset = []
    
    # Add working samples (label 0)
    for i in range(num_samples // 2):
        dataset.append({
            "code": working_samples[i % len(working_samples)],
            "label": 0
        })
    
    # Add buggy samples (label 1)
    for i in range(num_samples // 2):
        dataset.append({
            "code": buggy_samples[i % len(buggy_samples)],
            "label": 1
        })
    
    # Add remaining sample if odd number
    if num_samples % 2 == 1:
        dataset.append({
            "code": working_samples[0],
            "label": 0
        })
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {filename} with {len(dataset)} samples")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create training dataset (5000 samples)
    create_dataset("train_csharp_bugs.jsonl", 5000)
    
    # Create validation dataset (1000 samples)  
    create_dataset("valid_csharp_bugs.jsonl", 1000)
    
    print("Dataset generation complete!")