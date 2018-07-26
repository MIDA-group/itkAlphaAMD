
#ifndef Log_h
#define Log_h

#include <stdio.h>
#include <iostream>
#include <string>

class Log {
public:
    Log(const char* filename) {
        file = fopen(filename, "wb");
    }
    ~Log() {
        if(file) {
            fclose(file);
        }
    }

    bool isOpen() const { return file != nullptr; }
    
    void writeInteger(const char* label, long long value) {
        if(!file) return;
        
        if(label) {
            fprintf(file, "%s: %d\n", label, (int)value);
        } else {
            fprintf(file, "%d\n", (int)value);
        }
    }
    
    void writeFloat(const char* label, double value) {
        if(!file) return;
        
        if(label) {
            fprintf(file, "%s: %f\n", label, value);
        } else {
            fprintf(file, "%f\n", value);
        }
    }

    void writeString(const char* label, const char* value) {
        if(!file) return;
        
        if(label) {
            fprintf(file, "%s: %s\n", label, value);
        } else {
            fprintf(file, "%s\n", value);
        }
    }
    
    void writeString(const char* label, std::string value) {
        if(!file) return;
        
        if(label) {
            fprintf(file, "%s: %s\n", label, value.c_str());
        } else {
            fprintf(file, "%s\n", value.c_str());
        }
    }

    void writeString(std::string label, std::string value) {
        if(!file) return;
        
        fprintf(file, "%s: %s\n", label.c_str(), value.c_str());
    }
private:
    Log(const Log&) {}
    
    FILE* file;
};

#endif
