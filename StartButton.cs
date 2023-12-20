using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.SceneManagement;

public class StartButton : MonoBehaviour
{

    // 這個方法會在按鈕被點擊時被呼叫
    public void OnButtonClicked()
    {
        // 使用 SceneManager 切換場景
        keywordscene.r = false;
        StartCoroutine(LoadNextSceneAfterDelay(0.01f)); // Using StartCoroutine for delay

    } 

    private IEnumerator LoadNextSceneAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1);
    }

}

