#ifndef __MJ_ENV_H__
#define __MJ_ENV_H__

#include <iostream>
#include <string>
#include "mujoco.h"
#include <vector>
#include <GLFW/glfw3.h>


class Env{
public:
    friend std::ostream& operator<<(std::ostream& os, const Env& env);
    virtual std::string print() const;
protected:
    std::string info;

};

class PointMassEnv : public Env{
public:
    PointMassEnv(const char* envFile, const char* mjkey, bool view=false);
    virtual ~PointMassEnv();
    std::string print() const override;
    bool simulate(const std::vector<float> u);
    void step(std::vector<float>& x, const std::vector<float>& u);
    void get_x(std::vector<float> &x);

protected:
    bool view_;
    mjtNum _simend;


};


#endif
