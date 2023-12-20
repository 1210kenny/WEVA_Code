using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class AnimationControl : MonoBehaviour
{
    public Animator animator;
    string characterName;
    public int Index;
    public charcaterselection charcaterselection;

    private int index;
    private string[] characterNames = { "小花", "the3Dmodel","elsa"};
    public void Start()
    {
        //animator =GetComponent<Animator>();
     index = PlayerPrefs.GetInt("CharacterSelected");
     UnityEngine.Debug.Log("CharacterSelected index in animationcontroll: " + index);
     //charcaterselection.OnIndexConfirmed += HandleIndexConfirmed;
     
     //animator = GameObject.Find(characterNames[index]).GetComponent<Animator>();
    if(index==0){
        animator = GameObject.Find("小花").GetComponent<Animator>();
    }else if(index==1){
        animator = GameObject.Find("the3Dmodel").GetComponent<Animator>();
    }else if(index==2){
        animator = GameObject.Find("alice").GetComponent<Animator>();
    }else if(index==3){
        animator = GameObject.Find("evan").GetComponent<Animator>();
    }
    
     //}
         
    }
    
    

   
    public void set_face(string input){
        switch (input)
        {
            case "1":
                Set_Face_Default();
                Debug.Log("set_face:1");
                break;
            case "2":
                Set_Face_Fun();
                Debug.Log("set_face:2");
                break;
            case "3":
                Set_Face_Angry();
                Debug.Log("set_face:3");
                break;
            case "4":
                Set_Face_Sorrow();
                Debug.Log("set_face:4");
                break;
            case "5":
                Set_Face_Surprised();
                Debug.Log("set_face:5");
                break;
            case "6":
                Set_Face_Talk();
                Debug.Log("set_face:6");
                break;
            default:
                Set_Face_Default();
                Debug.Log("set_face:d");
                break;
        }
    }
    public void set_action(string input){
        switch (input)
        {
            case "1":
                Set_Body_Standing();
                Debug.Log("set_action:1");
                break;
            case "2":
                Set_Body_Bored();
                Debug.Log("set_action:2");
                break;
            case "3":
                Set_Body_Angry();
                Debug.Log("set_action:3");
                break;
            case "4":
                Set_Body_FormalBow();
                Debug.Log("set_action:4");
                break;
            case "5":
                Set_Body_InformalBow();
                Debug.Log("set_action:5");
                break;
            case "6":
                Set_Body_Sad();
                Debug.Log("set_action:6");
                break;
            case "7":
                Set_Body_Waving();
                Debug.Log("set_action:7");
                break;
            case "8":
                Set_Body_Yawn();
                Debug.Log("set_action:8");
                break;
            case "9":
                Set_Body_Happyidle();
                Debug.Log("set_action:9");
                break;
            case "10":
                Set_Body_ArmStretching();
                Debug.Log("set_action:10");
                break;
            case "11":
                Set_Body_Happy();
                Debug.Log("set_action:11");
                break;
            case "12":
                Set_Body_Excited();
                Debug.Log("set_action:12");
                break;
            default:
                Set_Face_Default();
                Debug.Log("set_action:d");
                break;
        }
    }
    public void Set_Face_Default()
    {
        animator.SetTrigger("Face_Default");
    }

    public void Set_Face_Fun()
    {
        animator.SetTrigger("Face_Fun");
    }

    public void Set_Face_Joy()
    {
        animator.SetTrigger("Face_Joy");
    }

    public void Set_Face_Surprised()
    {
        animator.SetTrigger("Face_Surprised");
    }

    public void Set_Face_Angry()
    {
        animator.SetTrigger("Face_Angry");
    }
    public void Set_Face_Sorrow()
    {
        animator.SetTrigger("Face_Sorrow");
    }

    public void Set_Body_Angry()
    {
        animator.SetTrigger("Body_Angry");
    }
    public void Set_Body_FormalBow()
    {
        animator.SetTrigger("Body_FormalBow");
    }
    public void Set_Body_Bored()
    {
        animator.SetTrigger("Body_Bored");
    }
    public void Set_Body_InformalBow()
    {
        animator.SetTrigger("Body_InformalBow");
    }
    public void Set_Body_Sad()
    {
        animator.SetTrigger("Body_Sad");
    }
    public void Set_Body_Waving()
    {
        animator.SetTrigger("Body_Waving");
    }
    public void Set_Body_Standing()
    {
        animator.SetTrigger("Body_Standing");
    }
    public void Set_Face_Talk()
    {
        animator.SetTrigger("Face_Talk");
    }
    public void Set_Body_Yawn()
    {
        animator.SetTrigger("Body_Yawn");
    }
    public void Set_Body_Happyidle()
    {
        animator.SetTrigger("Body_Happyidle");
    }
    public void Set_Body_ArmStretching()
    {
        animator.SetTrigger("Body_ArmStretching");
    }
    public void Set_Body_Happy()
    {
        animator.SetTrigger("Body_Happy");
    }
    public void Set_Body_Excited()
    {
        animator.SetTrigger("Body_Excited");
    }
}