#include "condaref.hpp"

#include "index.h"
#include "parameters.h"

/** @brief The CONDA index, its parameters, and the rows it inserts from */
struct CondaRef::Inner {
  efanna2e::Index<float> index;
  efanna2e::Parameters paras;
  const float *table;
  unsigned width;

  /**
   * @brief Build the index and its parameters as condaann's runner does, hop_count 2
   * @param setup the parameters
   * @param data the rows
   */
  Inner(const CondaSetup &setup, const float *data);
};

CondaRef::Inner::Inner(const CondaSetup &setup, const float *data)
    : index(setup.width, setup.count, efanna2e::L2, nullptr), table(data), width(setup.width) {
  // [1]
  paras.Set<unsigned>("L", setup.listn);
  paras.Set<unsigned>("R", setup.range);
  paras.Set<unsigned>("C", setup.prune);
  paras.Set<float>("alpha", setup.alpha);
  paras.Set<unsigned>("L_search", setup.listn);
  paras.Set<unsigned>("hop_count", 2);
  index.Init(paras);
}

CondaRef::CondaRef(const CondaSetup &setup, const float *data) : inner(new Inner(setup, data)) {
}

CondaRef::~CondaRef() = default;

void CondaRef::insert(const std::vector<unsigned> &tags) {
  // [1]
  std::vector<const float *> rows;
  for (unsigned tag : tags) {
    rows.push_back(inner->table + (size_t) tag * inner->width);
  }
  inner->index.BatchInsert_crng(tags, rows, inner->paras);
}

void CondaRef::remove(const std::vector<unsigned> &tags) {
  // [1]
  for (unsigned tag : tags) {
    inner->index.MarkDelete(tag);
  }
  inner->index.MultiProcess_crng(inner->paras);
  inner->index.ConsolidateDelete_conda(inner->paras);
}

std::vector<unsigned> CondaRef::edges(unsigned location) const {
  // [1]
  return inner->index.neighbors_of(location);
}

unsigned CondaRef::locate(unsigned tag) const {
  // [1]
  return inner->index.location_of(tag);
}

unsigned CondaRef::entry() const {
  // [1]
  return inner->index.entry_point();
}

unsigned CondaRef::locations() const {
  // [1]
  return inner->index.num_locations();
}
