#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <time.h>

/* =========================================================
 *  龙芯 AI 语音助手 —— 纯文本病历生成版 v3（专业强化）
 *  编译: gcc -o assistant assistant.c
 *
 *  修改点:
 *  1. 用 strrstr 从后往前抓 [病历开始]/[病历结束]，
 *     避免抓到 Prompt 自身的示例标记
 *  2. Prompt 加入 one-shot 示例 + 强制医学术语约束
 *  3. 纯英文文件名，避免编码问题
 * ========================================================= */

struct wav_header {
    char riff[4];
    int chunk_size;
    char wave[4];
    char fmt[4];
    int fmt_size;
    short audio_format;
    short num_channels;
    int sample_rate;
    int byte_rate;
    short block_align;
    short bits_per_sample;
    char data[4];
    int data_size;
};

void fix_wav_file(const char *raw_path, const char *wav_path) {
    FILE *raw_f = fopen(raw_path, "rb");
    if (!raw_f) return;
    fseek(raw_f, 0, SEEK_END);
    int pcm_size = ftell(raw_f);
    fseek(raw_f, 0, SEEK_SET);

    struct wav_header header;
    memcpy(header.riff, "RIFF", 4);
    header.chunk_size = pcm_size + 36;
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    header.fmt_size = 16;
    header.audio_format = 1;
    header.num_channels = 1;
    header.sample_rate = 16000;
    header.byte_rate = 32000;
    header.block_align = 2;
    header.bits_per_sample = 16;
    memcpy(header.data, "data", 4);
    header.data_size = pcm_size;

    FILE *wav_f = fopen(wav_path, "wb");
    fwrite(&header, sizeof(header), 1, wav_f);
    char buffer[4096];
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), raw_f)) > 0)
        fwrite(buffer, 1, bytes, wav_f);
    fclose(raw_f);
    fclose(wav_f);
}

void clean_text(const char *input, char *output) {
    int i = 0, j = 0, in_b = 0, in_t = 0;
    while (input[i]) {
        if (input[i] == '[') in_b = 1;
        else if (input[i] == ']') in_b = 0;
        else if (input[i] == '<') in_t = 1;
        else if (input[i] == '>') in_t = 0;
        else if (!in_b && !in_t && (unsigned char)input[i] > 31) output[j++] = input[i];
        i++;
    }
    output[j] = '\0';
}

/* 从后往前查找子串（标准 C 没有 strrstr） */
char *strrstr(const char *haystack, const char *needle) {
    char *result = NULL;
    char *p = (char *)haystack;
    while ((p = strstr(p, needle)) != NULL) {
        result = p;
        p++;
    }
    return result;
}

/* 从模型输出中提取最后一次出现的 [病历开始]...[病历结束] */
int extract_record(const char *raw, char *out, int out_size) {
    const char *start = strrstr(raw, "[病历开始]");
    const char *end = strrstr(raw, "[病历结束]");
    if (!start || !end || end <= start) return 0;
    start += strlen("[病历开始]");
    while (*start == '\n' || *start == '\r') start++;
    int len = end - start;
    if (len >= out_size) len = out_size - 1;
    strncpy(out, start, len);
    out[len] = '\0';
    int k = strlen(out) - 1;
    while (k > 0 && (out[k] == '\n' || out[k] == '\r' || out[k] == ' ')) out[k--] = '\0';
    return 1;
}

/* 安全过滤：去掉危险字符 */
void safe_filter(const char *src, char *dst, int dst_size) {
    int i = 0, j = 0;
    while (src[i] && j < dst_size - 1) {
        if (src[i] == '"' || src[i] == '\\' || src[i] == '$' || src[i] == '`')
            { i++; continue; }
        dst[j++] = src[i++];
    }
    dst[j] = '\0';
}

/* =========================================================
 *  核心：生成纯文本 A4 病历（ASCII 边框）
 * ========================================================= */
void save_txt_record(const char *record_raw, const char *patient_input) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char filename[256];
    snprintf(filename, sizeof(filename), "/home/richard/bingli_%04d%02d%02d_%02d%02d%02d.txt",
             t->tm_year+1900, t->tm_mon+1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

    FILE *f = fopen(filename, "w");
    if (!f) { printf("[系统] 无法创建病历文件\n"); return; }

    fprintf(f, "+==============================================================+\n");
    fprintf(f, "|                     门 诊 病 历                                |\n");
    fprintf(f, "+==============================================================+\n");
    fprintf(f, "|  病历编号 : %04d%02d%02d%02d%02d%02d                           |\n",
            t->tm_year+1900, t->tm_mon+1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    fprintf(f, "|  就诊日期 : %04d-%02d-%02d    时间 : %02d:%02d                    |\n",
            t->tm_year+1900, t->tm_mon+1, t->tm_mday, t->tm_hour, t->tm_min);
    fprintf(f, "|  科    室 : 全科门诊                                         |\n");
    fprintf(f, "|  医    生 : AI 辅助 / 需执业医师审核签字                     |\n");
    fprintf(f, "+--------------------------------------------------------------+\n");
    fprintf(f, "| 【主诉 / 患者自述】                                            |\n");
    fprintf(f, "|                                                                |\n");
    fprintf(f, "|   %s\n", patient_input);
    fprintf(f, "|                                                                |\n");
    fprintf(f, "+--------------------------------------------------------------+\n");
    fprintf(f, "| 【现病史 / 处理意见】                                          |\n");
    fprintf(f, "|                                                                |\n");

    char buf[4096];
    strncpy(buf, record_raw, sizeof(buf)-1);
    buf[sizeof(buf)-1] = '\0';
    char *line = strtok(buf, "\n");
    while (line) {
        fprintf(f, "|   %s\n", line);
        line = strtok(NULL, "\n");
    }

    fprintf(f, "|                                                                |\n");
    fprintf(f, "+--------------------------------------------------------------+\n");
    fprintf(f, "|                                                                |\n");
    fprintf(f, "|  医生签名 : ___________________    日期 : %04d-%02d-%02d          |\n",
            t->tm_year+1900, t->tm_mon+1, t->tm_mday);
    fprintf(f, "|                                                                |\n");
    fprintf(f, "+==============================================================+\n");
    fprintf(f, "|  声明：本病历由 AI 语音助手辅助生成，仅供执业医师参考，        |\n");
    fprintf(f, "|        不具有独立诊断和法律效力，须由执业医师签字确认生效。    |\n");
    fprintf(f, "+==============================================================+\n");

    fclose(f);
    printf("\n[系统] 病历已保存: %s\n", filename);
    printf("[系统] 查看: cat %s\n", filename);
}

/* =========================================================
 *  写入 prompt 临时文件（v3 专业强化版）
 * ========================================================= */
void write_prompt_file(const char *patient_input) {
    FILE *f = fopen("/tmp/prompt.txt", "w");
    if (!f) return;

    fprintf(f, "<|im_start|>system\n");
    fprintf(f, "你是一个门诊病历生成助手。用第三人称医学术语，禁止使用\"您建议可能考虑\"。\n");
    fprintf(f, "输出格式：\n");
    fprintf(f, "[病历开始]\n");
    fprintf(f, "主诉：...\n");
    fprintf(f, "现病史：...\n");
    fprintf(f, "初步诊断：...\n");
    fprintf(f, "处理意见：...\n");
    fprintf(f, "[病历结束]<|im_end|>\n");

    // user 只给输入，没有任何多余
    fprintf(f, "<|im_start|>user\n");
    fprintf(f, "%s<|im_end|>\n", patient_input);
    fprintf(f, "<|im_start|>assistant\n");

    fclose(f);
}

/* =========================================================
 *  main
 * ========================================================= */
int main() {
    const char *raw_file = "/tmp/rec.raw";
    const char *wav_file = "/tmp/rec.wav";

    printf("==================================================\n");
    printf("  龙芯 AI 语音助手【纯文本病历版 v3】\n");
    printf("==================================================\n");
    printf("  功能：语音 -> 识别 -> 生成 ASCII 病历 TXT\n");
    printf("==================================================\n");

    setbuf(stdout, NULL);  /* 关闭缓冲，确保 GUI 实时看到输出 */

    setenv("LD_LIBRARY_PATH", "/home/richard/sense_voice_demo/lib", 1);

    /* 杀掉之前残留的 assistant 进程，避免堆积 */
    {
        char buf[256];
        FILE *fp = popen("pidof assistant 2>/dev/null", "r");
        if (fp) {
            if (fgets(buf, sizeof(buf), fp)) {
                char *tok = strtok(buf, " \n");
                pid_t mypid = getpid();
                while (tok) {
                    pid_t p = atoi(tok);
                    if (p != mypid) { kill(p, SIGKILL); printf("[系统] 已终止旧进程 PID=%d\n", p); }
                    tok = strtok(NULL, " \n");
                }
            }
            pclose(fp);
        }
    }

    while (1) {
        printf("\n>>> 请说话描述症状... (按回车发送) <<<\n");

        pid_t pid = fork();
        if (pid == 0) {
            execlp("arecord", "arecord", "-D", "plughw:1,0", "-r", "16000", "-c", "1", "-f", "S16_LE", "-t", "raw", raw_file, NULL);
            exit(0);
        }

        while (getchar() != '\n'); 
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);

        printf("[系统] 正在封装音频并识别...\n");
        fix_wav_file(raw_file, wav_file);
        sync();

        char stt_cmd[1024];
        snprintf(stt_cmd, sizeof(stt_cmd), 
                 "/home/richard/sense_voice_demo/bin/sense-voice-main "
                 "-m /home/richard/sense_voice_demo/models/sense-voice-small-q4_k.gguf "
                 "-f %s -t 2 -np 2>/dev/null", wav_file);

        FILE *fp = popen(stt_cmd, "r");
        char buffer[1024], raw_res[4096] = {0}, final_res[4096] = {0};
        while (fgets(buffer, sizeof(buffer), fp)) if (strchr(buffer, '[')) strcat(raw_res, buffer);
        pclose(fp);

        clean_text(raw_res, final_res);
        if (strlen(final_res) < 2) { 
            printf("[系统] 没听清，请再说一次。\n"); 
            continue; 
        }

        printf("\n[你]: %s\n", final_res);
        printf("[系统] 正在生成病历...\n");
        fflush(stdout);

        /* Prompt 写入文件 + llama-cli -f 读取 */
        char safe_input[2048] = {0};
        safe_filter(final_res, safe_input, sizeof(safe_input));
        write_prompt_file(safe_input);

        char llama_cmd[1024];
        snprintf(llama_cmd, sizeof(llama_cmd),
                 "/home/richard/llama-cli "
                 "-m /home/richard/voice-assistant/models/qwen2.5-0.5b-instruct-q4_k_m.gguf "
                 "-c 512 -n 128 -t 2 --log-disable --no-display-prompt "
                 "-f /tmp/prompt.txt");

        FILE *llama_fp = popen(llama_cmd, "r");
        if (!llama_fp) {
            printf("[系统] 调用模型失败\n");
            continue;
        }
        
        char llama_buf[4096];
        char model_output[8192] = {0};
        while (fgets(llama_buf, sizeof(llama_buf), llama_fp)) {
            printf("%s", llama_buf);
            if (strlen(model_output) + strlen(llama_buf) < sizeof(model_output) - 1)
                strcat(model_output, llama_buf);
        }
        pclose(llama_fp);
        
        /* 用 strrstr 从后往前抓，精准提取模型生成的病历 */
        char record_content[4096] = {0};
        if (extract_record(model_output, record_content, sizeof(record_content))) {
            save_txt_record(record_content, final_res);
        } else {
            printf("\n[系统] 警告：模型未按格式输出，已保存原始文本。\n");
            save_txt_record(model_output, final_res);
        }

        printf("\n--------------------------------------------------\n");
    }
    return 0;
}
