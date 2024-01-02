using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class menulist : MonoBehaviour
{
    public inputChat inputChat;

    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void Return()
    {
        
    }
    public void Restart()
    {
        inputChat.Speaker.Mute();
        SceneManager.LoadScene(1);
    }
    public void Exit()
    {
        inputChat.Speaker.Mute();
        Application.Quit();
    }
    public void change()
    {
        inputChat.Speaker.Mute();
        SceneManager.LoadScene(2);
        
    }
    public void start()
    {
        inputChat.Speaker.Mute();
        SceneManager.LoadScene(0);
    }
}
