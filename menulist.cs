using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class menulist : MonoBehaviour
{
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
        SceneManager.LoadScene(1);
        
    }
    public void Exit()
    {
        Application.Quit();
    }
    public void change()
    {
        SceneManager.LoadScene(2);
        
    }
    public void start()
    {
        SceneManager.LoadScene(0);

    }
}
