(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(clear b4)
		(on-table b6)
		(clear b6)
		(on-table b8)
		(clear b8)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b5 b0)
		(in-tower b5)
		(on b1 b5)
		(in-tower b1)
		(on b7 b1)
		(in-tower b7)
		(on b9 b7)
		(in-tower b9)
		(on b2 b9)
		(in-tower b2)
		(on b3 b2)
		(in-tower b3)
		(on b4 b3)
		(in-tower b4)
		(blue b0)
		(blue b1)
		(blue b4)
		(red b5)
		(blue b7)
		(red b8)
		(red b7)
		(red b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b8 b1)) (not (on b8 b7)) (not (on b8 b4)))))
)