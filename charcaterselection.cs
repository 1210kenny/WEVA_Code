using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.IO;

public class charcaterselection : MonoBehaviour
{
    // Start is called before the first frame update
    private GameObject[] characterlist;
    public GameObject[] name;
    private int index;


 
     
    private void Start()
    {
        index=PlayerPrefs.GetInt("CharacterSelected");
       

        characterlist=new GameObject[transform.childCount];

        for(int i=0;i<transform.childCount;i++)
        {
            characterlist[i]=transform.GetChild(i).gameObject;
        }

        foreach(GameObject go in characterlist)
         go.SetActive(false);


        //role= index;
        //numberr=index;

        //first index
        if (characterlist[index])
        {
            characterlist[index].SetActive(true);
            if (index >= 0 && index < name.Length)
            {
                name[index].SetActive(true);
            }
            else{
                Debug.Log("index = " + index);
            }
        }
        //sharedValue = index;

    }
   

    public void ToggleLeft()
    {

        //off model
        characterlist[index].SetActive(false);
        name[index].SetActive(false);

        index--;
        if(index<0)
         index=characterlist.Length-1;

        //on model
        PlayerPrefs.SetInt("CharacterSelected",index);
        PlayerPrefs.Save(); 
        characterlist[index].SetActive(true);
        name[index].SetActive(true);

    }

    public void ToggleRight()
    {

        //off model
        characterlist[index].SetActive(false);
        name[index].SetActive(false);

        index++;
        if(index==characterlist.Length)
         index=0;

        //on model
        PlayerPrefs.SetInt("CharacterSelected",index);
        PlayerPrefs.Save(); 
        characterlist[index].SetActive(true);
        name[index].SetActive(true);

    }

    public void Comfirmbutton()
    {
        switch(index){
            case 0:
                try
                {
                    File.WriteAllText("NowAssistent.txt", "芬尼(Fanny)");
                }
                catch
                {
                    // 如果讀取文件失敗，創建一個新的文件
                    using (var fs = File.Create("NowAssistent.txt"))
                    {
                        fs.Close(); // 確保文件被關閉和釋放
                    }
                    File.WriteAllText("NowAssistent.txt", "芬尼(Fanny)");
                }
                break;
            case 1:
                try
                {
                    File.WriteAllText("NowAssistent.txt", "米拉(Mira)");
                }
                catch
                {
                    // 如果讀取文件失敗，創建一個新的文件
                    using (var fs = File.Create("NowAssistent.txt"))
                    {
                        fs.Close(); // 確保文件被關閉和釋放
                    }
                    File.WriteAllText("NowAssistent.txt", "米拉(Mira)");
                }
                break;
            case 2:
                try
                {
                    File.WriteAllText("NowAssistent.txt", "路比(Ruby)");
                }
                catch
                {
                    // 如果讀取文件失敗，創建一個新的文件
                    using (var fs = File.Create("NowAssistent.txt"))
                    {
                        fs.Close(); // 確保文件被關閉和釋放
                    }
                    File.WriteAllText("NowAssistent.txt", "路比(Ruby)");
                } 
                break;
            case 3:
                try
                {
                    File.WriteAllText("NowAssistent.txt", "艾迪(Eddie)");
                }
                catch
                {
                    // 如果讀取文件失敗，創建一個新的文件
                    using (var fs = File.Create("NowAssistent.txt"))
                    {
                        fs.Close(); // 確保文件被關閉和釋放
                    }
                    File.WriteAllText("NowAssistent.txt", "艾迪(Eddie)");
                } 
                break;
        }
        PlayerPrefs.SetInt("CharacterSelected",index);
        PlayerPrefs.Save(); 
           
        SceneManager.LoadScene("SampleScene");
      
    }
}
