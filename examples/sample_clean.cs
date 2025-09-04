using System;
using System.Collections.Generic;
using System.IO;

namespace CleanExample
{
    public class SafeCalculator
    {
        // Safe division with error handling
        public double Divide(double a, double b)
        {
            if (Math.Abs(b) < double.Epsilon)
            {
                throw new ArgumentException("Division by zero is not allowed");
            }
            return a / b;
        }
        
        // Safe string length check
        public int GetStringLength(string input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }
            return input.Length;
        }
        
        // Safe array access
        public int GetFirstElement(int[] array)
        {
            if (array == null || array.Length == 0)
            {
                throw new ArgumentException("Array cannot be null or empty");
            }
            return array[0];
        }
        
        // Proper event management
        public event Action<string> OnMessage;
        
        public void Subscribe()
        {
            OnMessage += HandleMessage;
        }
        
        public void Unsubscribe()
        {
            OnMessage -= HandleMessage;
        }
        
        private void HandleMessage(string message)
        {
            Console.WriteLine(message ?? "Empty message");
        }
        
        // Safe iteration
        public void ProcessItems(List<string> items)
        {
            if (items == null) return;
            
            for (int i = 0; i < items.Count; i++)
            {
                if (items[i] == "skip")
                {
                    continue;
                }
                Console.WriteLine(items[i] ?? "null item");
            }
        }
        
        // Proper resource disposal
        public string ReadFile(string path)
        {
            if (string.IsNullOrEmpty(path))
            {
                throw new ArgumentException("Path cannot be null or empty");
            }
            
            using (var reader = new StreamReader(path))
            {
                return reader.ReadToEnd();
            }
        }
        
        // Safe multiplication with overflow check
        public int MultiplyLarge(int a, int b)
        {
            try
            {
                return checked(a * b);
            }
            catch (OverflowException)
            {
                throw new OverflowException($"Multiplication of {a} and {b} causes overflow");
            }
        }
        
        // Safe collection modification
        public void RemoveEvenNumbers(List<int> numbers)
        {
            if (numbers == null) return;
            
            for (int i = numbers.Count - 1; i >= 0; i--)
            {
                if (numbers[i] % 2 == 0)
                {
                    numbers.RemoveAt(i);
                }
            }
        }
    }
}