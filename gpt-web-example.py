from transformers import AutoTokenizer, AutoModel
import torch
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="code_snippet", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields, "Code Embeddings Collection")

# Define the collection name
collection_name = "code_embeddings_py"

# Check if the collection exists
if utility.has_collection(collection_name):
    # Drop the existing collection
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' dropped.")

# Create the collection
collection = Collection("code_embeddings_py", schema)

# Example list of code snippets
code_snippets = [
    "public int Add(int a, int b) { return a + b; }",
    "public void PrintMessage(string message) { Console.WriteLine(message); }",
    "public bool IsEven(int number) { return number % 2 == 0; }",
    "public double CalculateArea(double radius) { return Math.PI * Math.Pow(radius, 2); }",
    "public string ReverseString(string input) { char[] charArray = input.ToCharArray(); Array.Reverse(charArray); return new string(charArray); }",
    "public int[] SortArray(int[] array) { Array.Sort(array); return array; }",
    "public bool IsPalindrome(string word) { string reversed = new string(word.Reverse().ToArray()); return word.Equals(reversed, StringComparison.OrdinalIgnoreCase); }",
    "public int Factorial(int n) { if (n <= 1) return 1; else return n * Factorial(n - 1); }",
    "public void Swap(ref int a, ref int b) { int temp = a; a = b; b = temp; }",
    "public int FindMax(int[] numbers) { return numbers.Max(); }",
    "public int Fibonacci(int n) { if (n <= 1) return n; else return Fibonacci(n - 1) + Fibonacci(n - 2); }",
    "public bool ContainsSubstring(string mainString, string subString) { return mainString.Contains(subString); }",
    "public DateTime GetCurrentDateTime() { return DateTime.Now; }",
    "public double ConvertCelsiusToFahrenheit(double celsius) { return (celsius * 9/5) + 32; }",
    "public void PrintArray<T>(T[] array) { foreach (var item in array) { Console.WriteLine(item); } }",
    "public bool IsPrime(int number) { if (number <= 1) return false; for (int i = 2; i <= Math.Sqrt(number); i++) { if (number % i == 0) return false; } return true; }",
    "public string ConcatenateStrings(string[] strings) { return string.Join(\" \", strings); }",
    "public int CountVowels(string input) { int count = 0; foreach (char c in input.ToLower()) { if (\"aeiou\".Contains(c)) count++; } return count; }",
    "public double CalculateDistance(double x1, double y1, double x2, double y2) { return Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2)); }",
    "public string GetDayOfWeek(DateTime date) { return date.DayOfWeek.ToString(); }"
];

# Generate embeddings
embeddings = [generate_embedding(code) for code in code_snippets]

# Insert embeddings and code snippets into Milvus
collection.insert([embeddings, code_snippets])

# Create an index on the embedding field
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # Inner Product
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)

# Load the collection into memory
collection.load()

# Function to search for code snippets
def search_code(query, top_k=2):
    query_embedding = generate_embedding(query)
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["code_snippet"]  # Specify the field to retrieve
    )
    for result in results[0]:
        print(f"Similarity Score: {result.score}")
        print(f"Code Snippet: {result.entity.get('code_snippet')}\n")

# Example query
search_code("what function is used to sort the elements of an array?")
