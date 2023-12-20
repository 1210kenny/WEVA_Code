using UnityEngine;
using UnityEngine.UI;

public class buttonmenu : MonoBehaviour
{
    public GameObject panelToOpen;
    void Start()
    {
        Button button = GetComponent<Button>();

        
        if (button != null)
        {
            button.onClick.AddListener(OpenOrClosePanel);
        }
    }

    void OpenOrClosePanel()
    {
        
        if (panelToOpen != null)
        {
            
            panelToOpen.SetActive(true);
            Time.timeScale = (0);
        }
    }
}







