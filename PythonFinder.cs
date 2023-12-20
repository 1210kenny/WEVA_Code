using System.Diagnostics;
using UnityEngine;

public class PythonFinder : MonoBehaviour
{
    public string FindPythonInterpreter()
    {
        string command;
        string argument;

        if (Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor)
        {
            command = "cmd.exe";
            argument = "/C where python";
        }
        else if (Application.platform == RuntimePlatform.OSXPlayer || Application.platform == RuntimePlatform.OSXEditor)
        {
            command = "/bin/bash";
            argument = "-c \"which python3\"";
        }
        else
        {
            UnityEngine.Debug.LogError("Unsupported platform");
            return null;
        }

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = command,
            Arguments = argument,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        Process process = new Process
        {
            StartInfo = startInfo
        };

        process.Start();
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();

        return output.Trim();  // Remove trailing newline characters
    }

    void Start()
    {
        string pythonPath = FindPythonInterpreter();
        UnityEngine.Debug.Log("Python Path: " + pythonPath);

    }
}
