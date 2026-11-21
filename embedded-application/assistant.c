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
 * 【文件导读 · 注释补充】
 *  职责：龙芯(LoongArch) 嵌入式终端上的全离线语音病历助手主程序。
 *  数据流：arecord 录音(裸 PCM) -> fix_wav_file 补 WAV 头
 *        -> popen 调 sense-voice-main 语音识别 -> clean_text 清洗
 *        -> popen 调 llama-cli(Qwen2.5-0.5B Q4_K_M) 生成病历
 *        -> extract_record 提取正文 -> save_txt_record 落盘 TXT。
 *  函数清单：
 *    fix_wav_file      RAW PCM -> WAV 封装（补 44 字节 RIFF 头）
 *    clean_text        STT 输出去方括号/尖括号标记与控制字符
 *    strrstr           自实现"从后往前找子串"（标准 C 没有）
 *    extract_record    抓最后一对 [病历开始]/[病历结束] 之间正文
 *    safe_filter       shell 危险字符过滤（防注入）
 *    save_txt_record   生成 ASCII 边框病历 TXT
 *    write_prompt_file 写 ChatML 提示词到 /tmp/prompt.txt
 *    main              录音 -> 识别 -> 生成 的无限主循环
 *  上层：由 medical_platform.py(GTK/Vte 终端) 拉起并实时展示输出，
 *        两者靠 stdout 文本与固定路径临时文件松耦合。
 * ========================================================= */

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

/* WAV 文件头（44 字节）：PCM16 单声道的最小 RIFF 布局，
 * 字段顺序与偏移必须严格符合 RIFF 规范，不能调换 */
struct wav_header {
    char riff[4];            /* "RIFF" 魔数 */
    int chunk_size;          /* 整个文件大小 - 8 */
    char wave[4];            /* "WAVE" 格式族 */
    char fmt[4];             /* "fmt " 子块标识（注意第 4 字节是空格） */
    int fmt_size;            /* PCM 固定为 16 */
    short audio_format;      /* 1 = 无压缩 PCM */
    short num_channels;      /* 单声道 */
    int sample_rate;         /* 16000 Hz，与 arecord 参数一致 */
    int byte_rate;           /* 每秒字节数 = sample_rate * block_align */
    short block_align;       /* 一帧字节数 = 通道数 * 位宽/8 */
    short bits_per_sample;   /* 16 bit 定点 */
    char data[4];            /* "data" 数据子块标识 */
    int data_size;           /* PCM 原始字节数 */
};

/**
 * @brief  把 arecord 输出的无头 RAW PCM 封装成标准 WAV（补 44 字节 RIFF 头）
 * @param  raw_path  输入：裸 PCM 文件（/tmp/rec.raw，S16_LE 单声道 16k）
 * @param  wav_path  输出：封装后的 WAV（/tmp/rec.wav），SenseVoice 只认 WAV
 * @return 无；raw 打不开时直接 return（wav 不会被创建，由调用方 continue 兜底）
 * @note   在 main 循环里"录音结束 -> STT 识别"之间调用；本函数写死的采样参数
 *         必须与 main 里 arecord 的命令行参数严格一致，否则识别端读到错误采样率。
 * @note   已知隐患：wav_f 未判空，磁盘满/权限异常时 fwrite(NULL) 会段错误。
 */
void fix_wav_file(const char *raw_path, const char *wav_path) {
    FILE *raw_f = fopen(raw_path, "rb");
    if (!raw_f) return;
    fseek(raw_f, 0, SEEK_END);
    int pcm_size = ftell(raw_f);   /* RAW 无头：文件总长就是 PCM 数据长 */
    fseek(raw_f, 0, SEEK_SET);     /* 回到文件头准备搬运数据 */

    struct wav_header header;
    memcpy(header.riff, "RIFF", 4);
    header.chunk_size = pcm_size + 36;  /* RIFF 规定 = 数据长 + 除本字段外的头 36 字节 */
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    header.fmt_size = 16;
    header.audio_format = 1;            /* 无压缩 PCM */
    header.num_channels = 1;            /* arecord -c 1 */
    header.sample_rate = 16000;         /* arecord -r 16000 */
    header.byte_rate = 32000;           /* 16000 采样/秒 x 2 字节/采样 */
    header.block_align = 2;             /* 16bit 单声道 = 2 字节/帧 */
    header.bits_per_sample = 16;        /* arecord -f S16_LE */
    memcpy(header.data, "data", 4);
    header.data_size = pcm_size;

    FILE *wav_f = fopen(wav_path, "wb");
    fwrite(&header, sizeof(header), 1, wav_f);  /* 先写头，再跟数据 */
    char buffer[4096];                          /* 4KB 一段边读边写，控制内存占用 */
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), raw_f)) > 0)
        fwrite(buffer, 1, bytes, wav_f);
    fclose(raw_f);
    fclose(wav_f);
}

/**
 * @brief  清洗 STT 输出文本：剥掉 [] 与 <> 包裹的内容，并丢弃不可见控制字符
 * @param  input   输入：SenseVoice 原始输出（含 [TRANSCRIBE_START] 等标记行）
 * @param  output  输出：清洗后的纯文本（调用方自备缓冲区，需保证容量足够）
 * @return 无（结果写入 output）
 * @note   逐字节状态机：in_b / in_t 分别跟踪方括号、尖括号两种区间；
 *         UTF-8 中文字节均 > 0x7F，不会被 >31 的判断误伤，只丢控制字符。
 */
void clean_text(const char *input, char *output) {
    int i = 0, j = 0, in_b = 0, in_t = 0;   /* in_b: 在[]内  in_t: 在<>内 */
    while (input[i]) {
        if (input[i] == '[') in_b = 1;
        else if (input[i] == ']') in_b = 0;
        else if (input[i] == '<') in_t = 1;
        else if (input[i] == '>') in_t = 0;
        else if (!in_b && !in_t && (unsigned char)input[i] > 31) output[j++] = input[i];  /* 只保留区间外的可打印字符 */
        i++;
    }
    output[j] = '\0';   /* 手动补结束符：output 是裸缓冲区 */
}

/* 从后往前查找子串（标准 C 没有 strrstr） */
/**
 * @brief  自实现 strrstr：返回 needle 在 haystack 中最后一次出现的位置
 * @param  haystack  被搜索的长串（模型完整输出）
 * @param  needle    目标子串（如 "[病历开始]"，必须非空）
 * @return 最后一次匹配的指针；从未匹配返回 NULL
 * @note   为什么不用 strstr：模型经常把 Prompt 里的示例标记复读一遍，
 *         只有"最后一对"标记才是本次真实生成的病历。
 */
char *strrstr(const char *haystack, const char *needle) {
    char *result = NULL;
    char *p = (char *)haystack;
    while ((p = strstr(p, needle)) != NULL) {
        result = p;
        p++;   /* 只前移 1 字节继续找：允许重叠匹配；needle 非空故不会死循环 */
    }
    return result;
}

/* 从模型输出中提取最后一次出现的 [病历开始]...[病历结束] */
/**
 * @brief  提取模型输出中最后一对 [病历开始]…[病历结束] 之间的病历正文
 * @param  raw       模型完整输出（可能含 Prompt 回显与多对标记）
 * @param  out       输出缓冲（提取结果，调用方提供）
 * @param  out_size  out 容量，用于截断保护
 * @return 1 = 提取成功；0 = 标记缺失或顺序颠倒（调用方走兜底分支）
 * @note   依赖 strrstr 取"最后一次出现"，绕开 Prompt 示例的干扰；
 *         提取后还会做"去首尾换行/空格"的排版收尾。
 */
int extract_record(const char *raw, char *out, int out_size) {
    const char *start = strrstr(raw, "[病历开始]");
    const char *end = strrstr(raw, "[病历结束]");
    if (!start || !end || end <= start) return 0;   /* end<=start: 标记不配对（如只复读了一次） */
    start += strlen("[病历开始]");
    while (*start == '\n' || *start == '\r') start++;   /* 跳过标记后紧随的空行 */
    int len = end - start;
    if (len >= out_size) len = out_size - 1;   /* 截断保护，给 '\0' 留 1 字节 */
    strncpy(out, start, len);
    out[len] = '\0';
    int k = strlen(out) - 1;
    while (k > 0 && (out[k] == '\n' || out[k] == '\r' || out[k] == ' ')) out[k--] = '\0';  /* 右去空白，病历更整洁 */
    return 1;
}

/* 安全过滤：去掉危险字符 */
/**
 * @brief  命令行安全过滤：删除 " \ $ ` 四类 shell 元字符，防语音内容注入命令
 * @param  src       输入：清洗后的语音识别文本
 * @param  dst       输出：过滤后文本
 * @param  dst_size  dst 容量（含 '\0'）
 * @return 无
 * @note   识别文本先经本函数，再由 write_prompt_file 落入 /tmp/prompt.txt；
 *         虽然主路径走 llama-cli -f 读文件，不直接拼命令行，但仍按
 *         "不信任外部输入"原则过滤，防后续维护者改用拼接方式时踩坑。
 */
void safe_filter(const char *src, char *dst, int dst_size) {
    int i = 0, j = 0;
    while (src[i] && j < dst_size - 1) {
        if (src[i] == '"' || src[i] == '\\' || src[i] == '$' || src[i] == '`')
            { i++; continue; }   /* 双引号/反斜杠/美元符/反引号：shell 元字符，直接吞掉 */
        dst[j++] = src[i++];
    }
    dst[j] = '\0';
}

/* =========================================================
 *  核心：生成纯文本 A4 病历（ASCII 边框）
 * ========================================================= */
/**
 * @brief  把模型生成的病历渲染成带 ASCII 边框的门诊病历 TXT 并落盘
 * @param  record_raw     病历正文（extract_record 已提取，多行、'\n' 分隔）
 * @param  patient_input  患者原始主诉（写入"主诉/患者自述"栏）
 * @return 无；文件创建失败时仅打印提示
 * @note   文件固定写到 /home/richard/ 下（换机部署需同步改路径）；
 *         文件名 bingli_YYYYMMDD_HHMMSS.txt = 病历编号，秒级时间戳天然唯一。
 *         正文多行用 strtok 逐行切分后补 "|   " 左边框。
 */
void save_txt_record(const char *record_raw, const char *patient_input) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);   /* localtime 非线程安全，本程序单线程使用无碍 */
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
    strncpy(buf, record_raw, sizeof(buf)-1);   /* strtok 会原地改写缓冲，record_raw 是 const，必须先拷贝 */
    buf[sizeof(buf)-1] = '\0';
    char *line = strtok(buf, "\n");
    while (line) {
        fprintf(f, "|   %s\n", line);   /* 逐行补左边框；超宽不自动折行，打印前需人工检查 */
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
/**
 * @brief  按 Qwen2.5 的 ChatML 模板把 system+user 两段写入 /tmp/prompt.txt
 * @param  patient_input  患者主诉（已经过 safe_filter）
 * @return 无；打开失败静默返回（llama-cli 会读到旧文件，属可接受降级）
 * @note   v3 强化点：system 段内置 one-shot 四段式格式（主诉/现病史/初步诊断/
 *         处理意见）并夹住 [病历开始]/[病历结束] 标记，与 extract_record 配套；
 *         文件结尾停在 <|im_start|>assistant\n，模型接续生成即为目标病历。
 *         llama-cli 通过 -f 参数整体读入该文件作为 prompt。
 */
void write_prompt_file(const char *patient_input) {
    FILE *f = fopen("/tmp/prompt.txt", "w");
    if (!f) return;

    fprintf(f, "<|im_start|>system\n");   /* 以下为 ChatML 的 system 段 */
    fprintf(f, "你是一个门诊病历生成助手。用第三人称医学术语，禁止使用\"您建议可能考虑\"。\n");  /* 负面词约束：压制口语化建议词 */
    fprintf(f, "输出格式：\n");
    fprintf(f, "[病历开始]\n");           /* one-shot 示例骨架，模型照此输出 */
    fprintf(f, "主诉：...\n");
    fprintf(f, "现病史：...\n");
    fprintf(f, "初步诊断：...\n");
    fprintf(f, "处理意见：...\n");
    fprintf(f, "[病历结束]<|im_end|>\n");

    // user 只给输入，没有任何多余
    fprintf(f, "<|im_start|>user\n");
    fprintf(f, "%s<|im_end|>\n", patient_input);
    fprintf(f, "<|im_start|>assistant\n");   /* 生成起点：模型从这里续写 */

    fclose(f);
}

/* =========================================================
 *  main
 * ========================================================= */
/**
 * @brief  主循环：录音 -> WAV 封装 -> SenseVoice 识别 -> 提文本 -> llama-cli 生成 -> 落盘
 * @return 永不返回（while(1)；由 GUI 的 Ctrl+C 或 SIGKILL 结束）
 * @note   与 medical_platform.py 的 GUI 靠"终端文本"松耦合：GUI 展示所有 stdout；
 *         三段外部能力（录音/识别/推理）全部经 fork+exec 或 popen 调外部进程，
 *         本进程自身只做编排与文本处理。
 */
int main() {
    const char *raw_file = "/tmp/rec.raw";    /* arecord 直写的裸 PCM */
    const char *wav_file = "/tmp/rec.wav";    /* 封装后喂给 SenseVoice 的 WAV */

    printf("==================================================\n");
    printf("  龙芯 AI 语音助手【纯文本病历版 v3】\n");
    printf("==================================================\n");
    printf("  功能：语音 -> 识别 -> 生成 ASCII 病历 TXT\n");
    printf("==================================================\n");

    setbuf(stdout, NULL);  /* 关闭缓冲，确保 GUI 实时看到输出 */

    setenv("LD_LIBRARY_PATH", "/home/richard/sense_voice_demo/lib", 1);  /* SenseVoice 私有动态库路径，popen 子进程继承 */

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
                    if (p != mypid) { kill(p, SIGKILL); printf("[系统] 已终止旧进程 PID=%d\n", p); }  /* 排除自己，防自杀 */
                    tok = strtok(NULL, " \n");
                }
            }
            pclose(fp);
        }
    }   /* 背景：GUI 每点一次"开始服务"都会 gcc+拉起新实例，先清场防多进程抢声卡 */

    while (1) {
        printf("\n>>> 请说话描述症状... (按回车发送) <<<\n");

        pid_t pid = fork();
        if (pid == 0) {
            /* 子进程：参数与 fix_wav_file 的头字段一一对应；
             * plughw:1,0 = 第 2 块声卡的插混设备（USB 麦克风常见序号） */
            execlp("arecord", "arecord", "-D", "plughw:1,0", "-r", "16000", "-c", "1", "-f", "S16_LE", "-t", "raw", raw_file, NULL);
            exit(0);   /* 只有 execlp 失败才会走到这里，正常情况被父进程杀掉 */
        }

        while (getchar() != '\n');   /* 阻塞等回车 = "说完了吗"的停止信号 */
        kill(pid, SIGKILL);          /* 强停录音：RAW 文件留成"录音到哪算哪" */
        waitpid(pid, NULL, 0);       /* 回收子进程，防僵尸 */

        printf("[系统] 正在封装音频并识别...\n");
        fix_wav_file(raw_file, wav_file);
        sync();   /* 强制落盘，防止 STT 子进程读到不完整文件 */

        char stt_cmd[1024];
        snprintf(stt_cmd, sizeof(stt_cmd),
                 "/home/richard/sense_voice_demo/bin/sense-voice-main "
                 "-m /home/richard/sense_voice_demo/models/sense-voice-small-q4_k.gguf "
                 "-f %s -t 2 -np 2>/dev/null", wav_file);   /* -t 2 双线程；-np 静默；日志丢弃只留结果 */

        FILE *fp = popen(stt_cmd, "r");
        char buffer[1024], raw_res[4096] = {0}, final_res[4096] = {0};
        while (fgets(buffer, sizeof(buffer), fp)) if (strchr(buffer, '[')) strcat(raw_res, buffer);  /* 只收带标记的结果行，过滤运行日志 */
        pclose(fp);

        clean_text(raw_res, final_res);
        if (strlen(final_res) < 2) {
            printf("[系统] 没听清，请再说一次。\n");   /* 空/单字节识别视为噪声 */
            continue;
        }

        printf("\n[你]: %s\n", final_res);
        printf("[系统] 正在生成病历...\n");
        fflush(stdout);

        /* Prompt 写入文件 + llama-cli -f 读取 */
        char safe_input[2048] = {0};
        safe_filter(final_res, safe_input, sizeof(safe_input));   /* 先过滤再入 prompt，双保险 */
        write_prompt_file(safe_input);

        char llama_cmd[1024];
        snprintf(llama_cmd, sizeof(llama_cmd),
                 "/home/richard/llama-cli "
                 "-m /home/richard/voice-assistant/models/qwen2.5-0.5b-instruct-q4_k_m.gguf "
                 "-c 512 -n 128 -t 2 --log-disable --no-display-prompt "
                 "-f /tmp/prompt.txt");   /* -c 512 上下文窗；-n 128 限生成长度控时延；-t 2 双线程 */

        FILE *llama_fp = popen(llama_cmd, "r");
        if (!llama_fp) {
            printf("[系统] 调用模型失败\n");
            continue;
        }

        char llama_buf[4096];
        char model_output[8192] = {0};
        while (fgets(llama_buf, sizeof(llama_buf), llama_fp)) {
            printf("%s", llama_buf);   /* 边收边打印，GUI 侧实时滚动 */
            if (strlen(model_output) + strlen(llama_buf) < sizeof(model_output) - 1)
                strcat(model_output, llama_buf);   /* 累积前先查容量，防 8KB 溢出 */
        }
        pclose(llama_fp);

        /* 用 strrstr 从后往前抓，精准提取模型生成的病历 */
        char record_content[4096] = {0};
        if (extract_record(model_output, record_content, sizeof(record_content))) {
            save_txt_record(record_content, final_res);   /* 主路径：提取到的四段式病历 */
        } else {
            printf("\n[系统] 警告：模型未按格式输出，已保存原始文本。\n");
            save_txt_record(model_output, final_res);   /* 兜底：宁可保存原始输出也不丢数据 */
        }

        printf("\n--------------------------------------------------\n");
    }
    return 0;
}
