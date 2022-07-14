class Node {
public:
    Node() {}
    map<char, Node*> children;
    bool isWord = false;   
};

class Trie {
private:
    Node* root;
public:
    Trie() {
        root = new Node();
    }
    
    ~Trie() {
        for(auto node : root->children) {
            delete node.second;
        }
        delete root;
    }
    
    void insert(string word) {
        Node* cur = root;
        for(char c : word) {
            if(cur->children.find(c) == cur->children.end()) {
                cur->children[c] = new Node(); 
            }
            cur = cur->children[c];
        }
        cur->isWord = true;
    }
    
    bool search(string word) {
        Node* cur = root;
        for(auto c : word) {
            if(cur->children.find(c) == cur->children.end()) {
                return false; 
            }
            cur = cur->children[c];
        }
        return cur->isWord;
    }
    
    bool startsWith(string prefix) {
        Node* cur = root;
        for(char c : prefix) {
            if(cur->children.find(c) == cur->children.end()) {
                return false; 
            }
            cur = cur->children[c];
        }
        return true;
    }
};

// create pointer object like:
// Trie* obj = new Trie();
// don't forget to delete this later