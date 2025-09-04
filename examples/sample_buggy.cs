using System;
using System.Collections.Generic;

namespace BuggyExample;

public class Calculator
{
    // Bug 1: Division by zero not handled
    public double Divide(double a, double b)
    {
        return a / b;  // Will crash if b is 0
    }
    
    // Bug 2: Null reference exception potential
    public int GetStringLength(string input)
    {
        return input.Length;  // Will crash if input is null
    }
    
    // Bug 3: Array index out of bounds
    public int GetFirstElement(int[] array)
    {
        return array[0];  // Will crash if array is empty
    }
    
    // Bug 4: Memory leak - event handler not unsubscribed
    public event Action<string> OnMessage;
    
    public void Subscribe()
    {
        OnMessage += HandleMessage;
        // Missing unsubscribe logic
    }
    
    private void HandleMessage(string message)
    {
        Console.WriteLine(message);
    }
    
    // Bug 5: Infinite loop potential
    public void ProcessItems(List<string> items)
    {
        int i = 0;
        while (i < items.Count)
        {
            if (items[i] == "skip")
            {
                continue;  // i never incremented, infinite loop
            }
            Console.WriteLine(items[i]);
            i++;
        }
    }
    
    // Bug 6: Resource not disposed properly
    public string ReadFile(string path)
    {
        var reader = new System.IO.StreamReader(path);
        return reader.ReadToEnd();
        // StreamReader not disposed, resource leak
    }
    
    // Bug 7: Integer overflow not handled
    public int MultiplyLarge(int a, int b)
    {
        return a * b;  // Can overflow without checking
    }
    
    // Bug 8: Concurrent modification exception potential
    public void RemoveEvenNumbers(List<int> numbers)
    {
        foreach (int num in numbers)
        {
            if (num % 2 == 0)
            {
                numbers.Remove(num);  // Modifying collection during iteration
            }
        }
    }
}
