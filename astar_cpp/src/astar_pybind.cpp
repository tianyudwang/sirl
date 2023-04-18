#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <list>
#include <unordered_map>                  // hash
#include <boost/heap/d_ary_heap.hpp>      // heap


namespace astar
{
  static constexpr float infCost = std::numeric_limits<float>::infinity();
  static constexpr float sqrtTwoCost = std::sqrt(2);
  
  template <class T>
  struct compareAStates
  {
    bool operator()(T* a1, T* a2) const
    {
      float f1 = a1->g + a1->h;
      float f2 = a2->g + a2->h;
      if( ( f1 >= f2 - 0.000001) && (f1 <= f2 + 0.000001) )
        return a1->g < a2->g; // if equal compare gvals
      return f1 > f2;
    }
  };
  
  struct AState; // forward declaration
  using PriorityQueue = boost::heap::d_ary_heap<AState*, boost::heap::mutable_<true>,
                        boost::heap::arity<2>, boost::heap::compare< compareAStates<AState> >>;
  
  struct AState
  {
    int x;
    int y;
    float g = infCost;
    float h;
    bool cl = false;
    AState* parent = nullptr;
    int u;                        // control from current node to parent node
    PriorityQueue::handle_type heapkey;
    AState( int x, int y ): x(x), y(y) {}
  };
  
  struct HashMap
  {
    HashMap(size_t sz)
    {
      if(sz < 1000000 ) hashMapV_.resize(sz);
      else useVec_ = false;
    }
    
    ~HashMap()
    {
      if(useVec_)
        for(auto it:hashMapV_)
          if(it)
            delete it;
      else
        for( auto it = hashMapM_.begin(); it != hashMapM_.end(); ++it )
          if(it->second)
            delete it->second;
    }
        
    AState*& operator[] (size_t n)
    {
      if(useVec_) return hashMapV_[n];
      else return hashMapM_[n];
    }
  private:
    std::vector<AState*> hashMapV_;
    std::unordered_map<size_t, AState*> hashMapM_;
    bool useVec_ = true;
  };

  typedef pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> pyArrayXd;
  typedef pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> pyArrayXi;

  void
    planOn2DGrid_cpp(
        const pybind11::detail::unchecked_reference<float, 3>& cMapData,
        const pybind11::detail::unchecked_reference<int, 1>& startData,
        const pybind11::detail::unchecked_reference<int, 1>& goalData,
        pybind11::detail::unchecked_mutable_reference<float, 2>& Q_batchData,
        pybind11::detail::unchecked_mutable_reference<float, 5>& dQdc_batchData,
        pybind11::detail::unchecked_mutable_reference<float, 3>& g_batchData,
        size_t xDim,
        size_t yDim,
        int k,
        float epsilon = 1.0)
  {

    size_t cMapLength = xDim*yDim;

    // Initialize the HashMap and Heap
    HashMap hm(cMapLength);
    PriorityQueue pq;

    // Initialize from goal state and search towards start state
    // This is because we want the g-val to be the optimal 
    // cost-to-go from current state to the goal
    AState *currNode_pt = new AState(goalData(0),goalData(1));
    currNode_pt->g = 0.0;
    currNode_pt->h = epsilon*std::sqrt( (goalData(0)-startData(0))*(goalData(0)-startData(0)) +
                                        (goalData(1)-startData(1))*(goalData(1)-startData(1)) );
    currNode_pt->cl = true;
    size_t indGoal = goalData(0) + xDim*goalData(1);    // column major
    hm[indGoal] = currNode_pt;
    
    // neighbors for 4-connected controls
    std::vector<std::vector<int>> neighbors = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

    int count = 0;
    while(true)
    { 
      // get g(x)
      g_batchData(k, currNode_pt->x, currNode_pt->y) = currNode_pt->g;
      // iterate over neighbor nodes
      for (int u = 0; u < neighbors.size(); u++) {
        int xNeighbor = currNode_pt->x + neighbors[u][0];
        int yNeighbor = currNode_pt->y + neighbors[u][1];
        if (xNeighbor < 0 || xNeighbor >= xDim || yNeighbor < 0 || yNeighbor >= yDim) continue;
        
        // initialize if never seen before
        size_t indNeighbor = xNeighbor + xDim*yNeighbor;
        AState*& child_pt = hm[indNeighbor];
        if( !child_pt )
          child_pt = new AState(xNeighbor,yNeighbor);
          hm[indNeighbor] = child_pt;
          
        // skip closed nodes
        if( child_pt->cl ) continue;

        // get control from neighbor nodes to current node
        int u_child;
        switch (u) {
          case 0: u_child = 2; break;
          case 1: u_child = 3; break;
          case 2: u_child = 0; break;
          case 3: u_child = 1; break; 
        }
          
        // Calculate cost
        float stageCost = cMapData(u, xNeighbor, yNeighbor);
        float costNeighbor = currNode_pt->g + stageCost;
          
        if (costNeighbor < child_pt->g) {
          //update the heuristic value
          if ( !std::isinf(child_pt->g) ) {
            // node seen before
            child_pt->g = costNeighbor;
            // increase == decrease with our comparator (!)
            pq.increase(child_pt->heapkey);
          }
          else {
            // ADD
            child_pt->h = epsilon*std::sqrt((xNeighbor-startData(0))*(xNeighbor-startData(0)) + 
                                            (yNeighbor-startData(1))*(yNeighbor-startData(1)));
            child_pt->g = costNeighbor;
            child_pt->heapkey = pq.push(child_pt);
          }
          child_pt->parent = currNode_pt;
          child_pt->u = u;
        }
      }
      

      // terminate if the start node and all its neighbors are in CLOSE
      bool terminate = true;
      size_t indStart = startData(0) + xDim*startData(1);
      AState*& startNode_pt = hm[indStart];
      if (!startNode_pt) {
        // continue loop if start is not expanded
        terminate = false;
      } 
      else {
        // continue loop if start is not in CLOSE
        if (!startNode_pt->cl) {
          terminate = false;
        } 
        else {
          // check if neighbor node is in CLOSE
          for (int u = 0; u < neighbors.size(); u++) {
            int xNeighbor = startNode_pt->x + neighbors[u][0];
            int yNeighbor = startNode_pt->y + neighbors[u][1];
            if (xNeighbor < 0 || xNeighbor >= xDim || yNeighbor < 0 || yNeighbor >= yDim) continue;
        
            size_t indNeighbor = xNeighbor + xDim*yNeighbor;
            AState*& neighbor_pt = hm[indNeighbor];
            if (!neighbor_pt) {
              terminate = false; 
              break;
            }
            if (!neighbor_pt->cl) {
              terminate = false; 
              break;
            }
          }
        }
      }
      if (terminate) {
        break;
      }

      currNode_pt = pq.top(); pq.pop(); // get element with smallest cost
      currNode_pt->cl = true;
      count++;
    }   

    // gather values for Q and dQdc
    // loop over the neighbors of start node 
    for (int u = 0; u < neighbors.size(); u++) {
      int xNeighbor = startData(0) + neighbors[u][0];
      int yNeighbor = startData(1) + neighbors[u][1];
      size_t indNeighbor = xNeighbor + xDim*yNeighbor;
      auto child_pt = hm[indNeighbor];       // child pt must exist

      // get Q(x_start, u)
      Q_batchData(k, u) = child_pt->g + cMapData(u, startData(0), startData(1));
      // get path from neighbor to goal
      dQdc_batchData(k, u, u, startData(0), startData(1)) = 1.;
      // continue tracing the path if child_pt exists and is not the goal
      while (child_pt->x != goalData(0) || child_pt->y != goalData(1)) { 
        dQdc_batchData(k, u, child_pt->u, child_pt->x, child_pt->y) = 1;
        child_pt = child_pt->parent;
      }
    }
  }

  void
  planOn2DGrid(const pybind11::array & cMap,
               const pybind11::array & start,
               const pybind11::array & goal,
               pybind11::array & Q_batch,
               pybind11::array & dQdc_batch,
               pybind11::array & g_batch,
               int k,
               float epsilon = 1.0) 
  {

    // Get references to the data and dimensions
    auto cMapData = cMap.unchecked<float,3>();
    auto startData = start.unchecked<int,1>();
    auto goalData = goal.unchecked<int,1>();
    auto Q_batchData = Q_batch.mutable_unchecked<float,2>();
    auto dQdc_batchData = dQdc_batch.mutable_unchecked<float,5>();
    auto g_batchData = g_batch.mutable_unchecked<float,3>();
    size_t xDim = cMapData.shape(1);
    size_t yDim = cMapData.shape(2);
    planOn2DGrid_cpp(cMapData,
        startData,
        goalData,
        Q_batchData,
        dQdc_batchData,
        g_batchData,
        xDim,
        yDim,
        k,
        epsilon);
  }

  void
  planBatch2DGrid(const pyArrayXd &cMap_batch,
                  const pyArrayXi &start_batch,
                  const pyArrayXi &goal_batch,
                  pyArrayXd &Q_batch,
                  pyArrayXd &dQdc_batch,
                  pyArrayXd &g_batch) 
  {
    // Get references to the data and dimensions
    auto cMapData_batch = cMap_batch.unchecked<4>();
    size_t batch_size = cMapData_batch.shape(0);

    // We need to avoid python operations in the threads, so need to put
    // indexed arrays in a cpp vector.
    std::vector<pybind11::detail::unchecked_reference<float, 3>> cMap_batch_vec;
    std::vector<pybind11::detail::unchecked_reference<int, 1>> start_batch_vec;
    std::vector<pybind11::detail::unchecked_reference<int, 1>> goal_batch_vec;
    std::vector<size_t> xDim_vec;
    std::vector<size_t> yDim_vec;

    for (int k = 0; k < batch_size; k++) {
      auto k_ellipsis = pybind11::make_tuple(k, pybind11::ellipsis());
      const pybind11::array& cMap_batch_k = cMap_batch[k_ellipsis];
      auto cMap_batchData = cMap_batch_k.unchecked<float,3>();
      cMap_batch_vec.push_back(cMap_batchData);

      xDim_vec.push_back(cMap_batchData.shape(1));
      yDim_vec.push_back(cMap_batchData.shape(2));

      const pybind11::array& start_batch_k = start_batch[k_ellipsis];
      auto start_batchData = start_batch_k.unchecked<int, 1>();
      start_batch_vec.push_back(start_batchData);

      const pybind11::array& goal_batch_k = goal_batch[k_ellipsis];
      auto goal_batchData = goal_batch_k.unchecked<int, 1>();
      goal_batch_vec.push_back(goal_batchData);
    }
    pybind11::array& Q_batch_aref = Q_batch;
    auto Q_batchData = Q_batch_aref.mutable_unchecked<float,2>();
    pybind11::array& dQdc_batch_aref = dQdc_batch;
    auto dQdc_batchData = dQdc_batch_aref.mutable_unchecked<float,5>();
    pybind11::array& g_batch_aref = g_batch;
    auto g_batchData = g_batch_aref.mutable_unchecked<float,3>();

#pragma omp parallel for
    for (int k = 0; k < batch_size; ++k) {
      planOn2DGrid_cpp(
          cMap_batch_vec[k],
          start_batch_vec[k],
          goal_batch_vec[k],
          Q_batchData,
          dQdc_batchData,
          g_batchData,
          xDim_vec[k],
          yDim_vec[k],
          k);
    }
  }
}


PYBIND11_MODULE(astar_pybind, m)
{
  m.doc() = "Pybind11 A* plugin";
  
  m.def("planBatch2DGrid", &astar::planBatch2DGrid,
      "Plan on a batch of 2D grids using A*.\n\n"
      "Input:\n"
      "\tcMap_batch:\n"
      "\tstart_batch:\n"
      "\tgoal_batch:\n",
      "\tQ_batch:\n",
      "\tdQdc_batch:\n",
      "\tg_batch:\n",
      pybind11::arg("cMap_batch"),
      pybind11::arg("start_batch"),
      pybind11::arg("goal_batch"),
      pybind11::arg("Q_batch"),
      pybind11::arg("dQdc_batch"),
      pybind11::arg("g_batch"));

  m.attr("__version__") = "dev";
}
