package com.example.llama

import android.llama.cpp.LLamaAndroid
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

class MainViewModel(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0
    }

    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(listOf("Initializing..."))
        private set

    var message by mutableStateOf("")
        private set

    override fun onCleared() {
        super.onCleared()

        viewModelScope.launch {
            try {
                llamaAndroid.unload()
            } catch (exc: IllegalStateException) {
                messages += exc.message!!
            }
        }
    }

    // --- 在 MainViewModel 类中添加这两个变量 ---
    private val systemPrompt = "<|im_start|>system\n你是一个智能助手。<|im_end|>\n"
    private var isFirstTurn = true

    // --- 替换原有的 send() 函数 ---
    fun send() {
        val userText = message
        message = "" // 清空输入框

        // 更新 UI 显示 (给用户看纯文本)
        messages += "User: $userText"
        messages += "" // 占位符，等待 AI 回复

        // 构建发给模型的 Prompt (带 Qwen 标签)
        val promptBuilder = StringBuilder()
        if (isFirstTurn) {
            promptBuilder.append(systemPrompt)
            isFirstTurn = false
        }

        // Qwen 格式核心：
        promptBuilder.append("<|im_start|>user\n")
        promptBuilder.append(userText)
        promptBuilder.append("<|im_start|>assistant\n")

        val finalPrompt = promptBuilder.toString()

        viewModelScope.launch {
            // 发送带标签的文本给底层 C++
            llamaAndroid.send(finalPrompt)
                .catch {
                    Log.e(tag, "send() failed", it)
                    messages += it.message!!
                }
                .collect {
                    // 将回复拼接到最后一条消息上
                    messages = messages.dropLast(1) + (messages.last() + it)
                }
        }
    }

    fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) {
        viewModelScope.launch {
            try {
                val start = System.nanoTime()
                val warmupResult = llamaAndroid.bench(pp, tg, pl, nr)
                val end = System.nanoTime()

                messages += warmupResult

                val warmup = (end - start).toDouble() / NanosPerSecond
                messages += "Warm up time: $warmup seconds, please wait..."

                if (warmup > 5.0) {
                    messages += "Warm up took too long, aborting benchmark"
                    return@launch
                }

                messages += llamaAndroid.bench(512, 128, 1, 3)
            } catch (exc: IllegalStateException) {
                Log.e(tag, "bench() failed", exc)
                messages += exc.message!!
            }
        }
    }

    fun load(pathToModel: String) {
        viewModelScope.launch {
            try {
                llamaAndroid.load(pathToModel)
                messages += "Loaded $pathToModel"
            } catch (exc: IllegalStateException) {
                Log.e(tag, "load() failed", exc)
                messages += exc.message!!
            }
        }
    }

    fun updateMessage(newMessage: String) {
        message = newMessage
    }

    fun clear() {
        messages = listOf()
        isFirstTurn = true // 新增：重置对话轮次状态
    }

    fun log(message: String) {
        messages += message
    }
}
