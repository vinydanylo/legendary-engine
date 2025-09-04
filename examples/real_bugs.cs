// Example 1: Collection modification during iteration
public void RemoveEvens(List<int> nums) { foreach (var n in nums) { if (n % 2 == 0) nums.Remove(n); } }

// Example 2: Potential infinite loop  
public void BadLoop(int count) { for (int i = 1; i <= count; i++) { if (i == 5) i = 1; } }

// Example 3: Null reference
public int GetLength(string s) { return s.Length; }

// Example 4: Division by zero
public double Calc(int a, int b) { return a / b; }

// Example 5: Array bounds
public int First(int[] arr) { return arr[0]; }