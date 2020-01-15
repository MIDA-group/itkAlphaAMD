
#ifndef GRIDLOCK_H
#define GRIDLOCK_H

#include <mutex>
#include <vector>

class GridLock
{
public:
    void SetLockDensity(unsigned int density)
    {
        this->m_LockDensity = density;
    }
    
    void SetCount(unsigned int count)
    {
        this->m_Count = count;
    }

    void Initialize()
    {
        this->m_Locks.reset(new std::mutex[this->m_Count/this->m_LockDensity]);
        //this->m_Locks.clear();
        //this->m_Locks.reserve(m_Count);
        //this->m_Locks.resize(m_Count);
        //for(unsigned int i = 0; i < this->m_Count; ++i)
        //{
            //std::mutex lck;
            //this->m_Locks.emplace_back(std::move(lck));
        //}
    }

    void Lock(int elementIndex)
    {
        assert(elementIndex >= 0 && elementIndex < this->m_Count);

        this->m_Locks[elementIndex / this->m_Count].lock();
    }

    void Unlock(int elementIndex)
    {
        assert(elementIndex >= 0 && elementIndex < this->m_Count);

        this->m_Locks[elementIndex / this->m_Count].unlock();
    }
private:
    std::unique_ptr<std::mutex[]> m_Locks;
    unsigned int m_LockDensity;
    unsigned int m_Count;
};

#endif
